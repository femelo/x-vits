"""Train the stage 4b prominence student (head on the frozen text encoder).

    python -m piper_train.prosody.train_prominence \\
        --checkpoint <stage2.ckpt> --prosody prosody_all.pt \\
        --dataset piper_s2/dataset.jsonl --eval-set dev_set.jsonl --eval-set test_set.jsonl \\
        --config config.json [--marked metadata_marked.csv ...] --out-dir prominence

Targets: acoustic word prominence from the recordings (prominence.py), score
>= --z-thr -> 1; words marked *like this* in --marked files are forced to 1.
Validation: held-out recordings' own acoustic labels (AUC, top-1 agreement).
"""

import argparse
import json
import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..eval.phonemize import PiperPhonemizer
from ..eval.synthesize import load_model
from .emphasis import parse_marked, token_word_index
from .predictor import text_hidden
from .prominence import ProminenceHead, pool_words, vowel_duration_means, word_prominence

_LOG = logging.getLogger("train_prominence")


def auc(scores, labels):
    s, y = np.asarray(scores), np.asarray(labels)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--prosody", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--eval-set", type=Path, action="append", required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--marked", type=Path, action="append", default=[], help="id|...|*marked* text files")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--z-thr", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1234)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); torch.manual_seed(a.seed)

    mg = load_model(a.checkpoint, a.device, False).model_g.eval().requires_grad_(False)
    pz = PiperPhonemizer(a.config)
    inv = {v: k for k, v in pz.id_map.items()}
    U = torch.load(a.prosody, weights_only=False)["utts"]
    marked = {}
    for f in a.marked:
        for l in open(f, encoding="utf-8"):
            q = l.rstrip("\n").split("|"); marked[q[0]] = q[-1]

    def build(rows):
        out = []
        for utt, r in rows:
            plain, flags = parse_marked(marked.get(utt, r["text"]))
            ids, tw = token_word_index(plain, pz)
            if ids != r["phoneme_ids"]:
                continue
            sc = word_prominence(U[utt], ids, tw, inv, vmeans)
            y = np.where(np.isnan(sc), -1, (sc >= a.z_thr).astype(int))
            y = np.where(np.array(flags), 1, y)
            out.append((utt, ids, np.array(tw), y, sc, plain))
        return out

    train_rows = [(Path(r["audio_path"]).stem, r) for r in map(json.loads, open(a.dataset))]
    vmeans = vowel_duration_means(U, [(u, r["phoneme_ids"]) for u, r in train_rows], inv)
    val_rows = [(r["utt"], r) for f in a.eval_set for r in map(json.loads, open(f))]
    assert not {u for u, _ in val_rows} & {u for u, _ in train_rows}
    train, val = build(train_rows), build(val_rows)
    _LOG.info("train %d, val %d sentences; positive word rate %.2f", len(train), len(val),
              np.mean(np.concatenate([y[y >= 0] for *_, y, _, _ in train])))

    head = ProminenceHead(text_channels=mg.enc_p.hidden_channels).to(a.device)
    _LOG.info("student parameters: %.2fM", sum(q.numel() for q in head.parameters()) / 1e6)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))

    def forward(batch):
        n = max(len(b[1]) for b in batch)
        ids = torch.zeros(len(batch), n, dtype=torch.long)
        for i, b in enumerate(batch):
            ids[i, : len(b[1])] = torch.LongTensor(b[1])
        lengths = torch.LongTensor([len(b[1]) for b in batch]).to(a.device)
        x, mask = text_hidden(mg, ids.to(a.device), lengths)
        tl = head(x, mask)
        words = []
        for i, b in enumerate(batch):
            tw = torch.from_numpy(b[2]).to(a.device)
            words.append(pool_words(tl[i, : len(b[1])], tw, len(b[3])))
        return words

    def evaluate():
        head.eval()
        S, Y, top1, base_last, base_rand = [], [], [], [], []
        with torch.no_grad():
            for i in range(0, len(val), 32):
                for b, wl in zip(val[i : i + 32], forward(val[i : i + 32])):
                    y, sc = b[3], b[4]
                    ok = y >= 0
                    S += wl.cpu().numpy()[ok].tolist(); Y += y[ok].tolist()
                    if ok.sum() >= 2 and not np.all(np.isnan(sc)):
                        best = int(np.nanargmax(sc))
                        pred = wl.cpu().numpy(); pred[~ok] = -1e9
                        top1.append(int(np.argmax(pred)) == best)
                        base_last.append(best == int(np.nonzero(ok)[0][-1]))
                        base_rand.append(1 / ok.sum())
        head.train()
        return auc(S, Y), float(np.mean(top1)), float(np.mean(base_last)), float(np.mean(base_rand))

    v = evaluate()
    _LOG.info("untrained: AUC %.3f top1 %.2f | baselines: last word %.2f, random %.2f", *v)
    best, step = -1.0, 0
    while step < a.steps:
        random.shuffle(train)
        for i in range(0, len(train) - a.batch_size + 1, a.batch_size):
            batch = train[i : i + a.batch_size]
            wls = forward(batch)
            logits = torch.cat([wl[torch.from_numpy(b[3] >= 0).to(a.device)] for wl, b in zip(wls, batch)])
            target = torch.cat([torch.from_numpy(b[3][b[3] >= 0]).float() for b in batch]).to(a.device)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sched.step(); step += 1
            if step % 250 == 0 or step == a.steps:
                v = evaluate()
                tag = ""
                if v[0] > best:
                    best, tag = v[0], " *"
                    torch.save({"state_dict": head.state_dict(), "kw": head.kw, "z_thr": a.z_thr, "step": step, "val_auc": v[0]}, a.out_dir / "best.pt")
                _LOG.info("step %d loss %.4f | val AUC %.3f top1 %.2f%s", step, loss.item(), v[0], v[1], tag)
            if step >= a.steps:
                break
    _LOG.info("done, best val AUC %.3f", best)


if __name__ == "__main__":
    main()
