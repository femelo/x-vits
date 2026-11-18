"""Train the stage 3 flow-matching prosody predictor against a frozen
(stage 2) VITS checkpoint.

Only the predictor is trained; the text encoder is run under no_grad to give
the conditioning.  Targets come from prosody.pt (stage 1 extraction).

  python -m piper_train.prosody.train_predictor \
      --checkpoint <stage2.ckpt> --prosody data/ljspeech/piper/prosody.pt \
      --dataset data/ljspeech/piper_prosody/dataset.jsonl \
      --eval-set .../eval_set.jsonl --out-dir runs/s3
"""

import argparse
import json
import logging
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

from ..eval.synthesize import load_model
from .predictor import ProsodyPredictor, n_params, text_hidden

_LOG = logging.getLogger("train_predictor")


def load_rows(path: Path, utts: dict, key: str):
    rows = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        utt = r["utt"] if key == "utt" else Path(r["audio_path"]).stem
        if utt in utts:
            rows.append((utt, r["phoneme_ids"]))
    return rows


def last_word_mask(pid, space_id, stop_ids):
    """1 for the tokens of the last word (after the last space, up to but
    excluding trailing punctuation/EOS), else 0."""
    n = len(pid)
    end = n
    while end > 0 and pid[end - 1] in stop_ids:
        end -= 1
    start = max((i + 1 for i in range(end) if pid[i] == space_id), default=0)
    m = np.zeros(n, np.float32)
    m[start:end] = 1.0
    return m


def make_batch(rows, targets, device, styles=None, p_drop=0.0, emph=None, e_drop=0.0):
    """styles: {utt: style index} (None = unconditional predictor); each
    label is dropped to 0 ("unspecified") with probability p_drop.
    emph: {utt: per-token emphasis}; zeroed per utterance with prob e_drop."""
    n = max(len(ids) for _, ids in rows)
    ids = torch.zeros(len(rows), n, dtype=torch.long)
    y = torch.zeros(len(rows), 4, n)
    lengths = torch.LongTensor([len(i) for _, i in rows])
    for b, (utt, pid) in enumerate(rows):
        ids[b, : len(pid)] = torch.LongTensor(pid)
        tgt = targets[utt]
        assert tgt.shape[0] == len(pid), utt
        y[b, :, : len(pid)] = torch.from_numpy(tgt).t()
    g = None
    if styles is not None:
        g = torch.LongTensor([0 if random.random() < p_drop else styles[u] for u, _ in rows]).to(device)
    f = None
    if emph is not None:
        a0 = next(iter(emph.values()))
        n_e = 1 if a0.ndim == 1 else a0.shape[0]
        f = torch.zeros(len(rows), n_e, n)
        for b, (utt, pid) in enumerate(rows):
            if random.random() >= e_drop:
                f[b, :, : len(pid)] = torch.from_numpy(emph[utt]).reshape(n_e, -1)
        f = f.to(device)
    return ids.to(device), lengths.to(device), y.to(device), g, f


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--prosody", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--eval-set", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--wn-layers", type=int, default=8)
    p.add_argument("--attn-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--style-regex",
        help="condition on an utterance style label: regex whose group 1 extracts it "
        r"from the utterance name, e.g. '^ex\d+_([a-z]+)_' for Expresso",
    )
    p.add_argument("--pos-features", action="store_true",
                   help="give the predictor each token's distance to sentence start/end")
    p.add_argument("--question-final-weight", type=float, default=1.0,
                   help="loss weight on the last word of questions (sentences with '?')")
    p.add_argument("--question-oversample", type=int, default=1,
                   help="repeat questions this many times per epoch")
    p.add_argument("--config", type=Path, help="voice config.json (phoneme map) for the "
                   "question options; default: <dataset dir>/config.json")
    p.add_argument("--emphasis", type=Path, help="emphasis.pt (stage 4 word emphasis per token)")
    p.add_argument("--emphasis-dropout", type=float, default=0.1)
    p.add_argument("--emphasis-weight", type=float, default=1.0,
                   help="loss weight on tokens of emphasised words")
    p.add_argument("--style-dropout", type=float, default=0.2,
                   help="probability of training a label as 'unspecified'")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    vits = load_model(args.checkpoint, args.device, False)
    model_g = vits.model_g.eval().requires_grad_(False)

    pros = torch.load(args.prosody, weights_only=False)
    utts = pros["utts"]
    logd = np.log1p(np.concatenate([u["dur"] for u in utts.values()]).astype(np.float64))
    stats = dict(pros["stats"], logdur_mean=float(logd.mean()), logdur_std=float(logd.std()))

    train = load_rows(args.dataset, utts, "audio")
    val = load_rows(args.eval_set, utts, "utt")
    assert not {u for u, _ in val} & {u for u, _ in train}, "eval utterances in training set"
    _LOG.info("train %d, val %d utterances", len(train), len(val))

    style_of, styles = None, None
    if args.style_regex:
        rx = re.compile(args.style_regex)
        names = {u: rx.search(u).group(1) for u, _ in train + val}
        styles = sorted(set(names.values()))
        style_of = {u: 1 + styles.index(n) for u, n in names.items()}
        _LOG.info("styles: %s", {k: sum(n == k for n in names.values()) for k in styles})

    emph = None
    if args.emphasis:
        emph = torch.load(args.emphasis, weights_only=False)["emphasis"]
        _LOG.info("emphasis: %d training utterances with marks",
                  sum(1 for u, _ in train if emph[u].any()))

    n_feats = 0
    if emph is not None:
        a0 = next(iter(emph.values()))
        n_feats = 1 if a0.ndim == 1 else a0.shape[0]
    pred = ProsodyPredictor(
        stats,
        styles,
        n_feats,
        text_channels=model_g.enc_p.hidden_channels,
        hidden=args.hidden,
        n_wn_layers=args.wn_layers,
        n_attn_layers=args.attn_layers,
        p_dropout=args.dropout,
        pos_features=args.pos_features,
    ).to(args.device)
    _LOG.info("predictor parameters: %.2fM", n_params(pred))
    targets = {u: pred.encode_targets(utts[u]) for u, _ in train + val}

    opt = torch.optim.AdamW(pred.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / args.warmup)
        * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)),
    )

    qw = None
    if args.question_final_weight != 1.0 or args.question_oversample > 1:
        pmap = json.load(open(args.config or args.dataset.parent / "config.json", encoding="utf-8"))["phoneme_id_map"]
        q_id, sp_id = pmap["?"][0], pmap[" "][0]
        stop = {pmap[k][0] for k in ("?", "!", ".", ",", "$", "_", '"') if k in pmap}
        qw = {}
        for u, pid in train:
            if q_id in pid:
                qw[u] = 1.0 + (args.question_final_weight - 1.0) * last_word_mask(pid, sp_id, stop)
        _LOG.info("questions in training: %d / %d (final-word weight %.1f, oversample x%d)",
                  len(qw), len(train), args.question_final_weight, args.question_oversample)
        train = train + [r for r in train if r[0] in qw] * (args.question_oversample - 1)

    def run(rows, p_drop=0.0, weighted=False):
        ids, lengths, y, g, f = make_batch(
            rows, targets, args.device, style_of, p_drop, emph, args.emphasis_dropout if p_drop else 0.0
        )
        x, x_mask = text_hidden(model_g, ids, lengths)
        w = None
        if weighted and qw is not None and args.question_final_weight != 1.0:
            w = torch.ones(len(rows), 1, ids.size(1))
            for b, (u, pid) in enumerate(rows):
                if u in qw:
                    w[b, 0, : len(pid)] = torch.from_numpy(qw[u])
        if weighted and emph is not None and args.emphasis_weight != 1.0:
            w = torch.ones(len(rows), 1, ids.size(1)) if w is None else w
            for b, (u, pid) in enumerate(rows):
                flag = emph[u].reshape(-1, len(pid))[0]
                w[b, 0, : len(pid)] *= torch.from_numpy(1.0 + (args.emphasis_weight - 1.0) * flag)
        if w is not None:
            w = w.to(args.device)
        return pred.loss(x, y, x_mask, style=g, weight=w, feats=f)

    @torch.no_grad()
    def validate():
        pred.eval()
        with torch.random.fork_rng(devices=[args.device] if args.device != "cpu" else []):
            torch.manual_seed(0)
            ls = [run(val[i : i + 50]).item() for _ in range(4) for i in range(0, len(val), 50)]
        pred.train()
        return float(np.mean(ls))

    best, step, t0, acc = float("inf"), 0, time.time(), []
    pred.train()
    while step < args.steps:
        random.shuffle(train)
        for i in range(0, len(train) - args.batch_size + 1, args.batch_size):
            loss = run(train[i : i + args.batch_size], args.style_dropout, weighted=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pred.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            acc.append(loss.item())
            if step % 200 == 0:
                _LOG.info("step %d loss %.4f lr %.2e %.1f it/s", step, np.mean(acc),
                          sched.get_last_lr()[0], 200 / (time.time() - t0))
                acc, t0 = [], time.time()
            if step % args.val_every == 0 or step == args.steps:
                v = validate()
                tag = ""
                if v < best:
                    best, tag = v, " *"
                    pred.save(args.out_dir / "best.pt", {"step": step, "val": v})
                pred.save(args.out_dir / "last.pt", {"step": step, "val": v})
                _LOG.info("step %d val %.4f%s", step, v, tag)
            if step >= args.steps:
                break
    _LOG.info("done, best val %.4f", best)


if __name__ == "__main__":
    main()
