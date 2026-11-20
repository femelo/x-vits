"""Style-conditioned stochastic duration predictor (SDP).

Fine-tunes only the voice's own SDP (``model_g.dp``) to take an utterance
style label, using the stage 1 durations (prosody.pt) as targets, exactly the
quantity the SDP is trained on inside VITS (frames per token, from MAS).  The
rest of the voice stays frozen, so nothing else about it changes.

The style enters as a learned vector added where a speaker embedding would
(``SDP.forward(h_add=...)``).  The embedding is zero-initialised, so at step 0
every style reproduces the original SDP; index 0 = "unspecified" (trained via
label dropout), which keeps unlabelled use working.

    python -m piper_train.prosody.train_dp_style \\
        --checkpoint <stage2.ckpt> --prosody prosody_all.pt \\
        --dataset piper_s2/dataset.jsonl --eval-set dev_set.jsonl \\
        --style-regex '^ex\\d+_([a-z]+)_' --out-dir dp_style

Writes ``best.pt`` / ``last.pt``: {"dp": state_dict, "style": state_dict,
"styles": [...]}; load at synthesis with ``load_style_dp``.
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
from torch import nn

from ..eval.synthesize import load_model
from .predictor import text_hidden
from .train_predictor import load_rows

_LOG = logging.getLogger("train_dp_style")


class DpCond(nn.Module):
    """Style embedding (+ optional per-token emphasis projection) -> SDP h_add."""

    def __init__(self, n_styles, channels, emphasis=False):
        """emphasis: False, True (1 channel) or the number of emphasis channels."""
        super().__init__()
        self.style = nn.Embedding(n_styles + 1, channels)
        nn.init.zeros_(self.style.weight)
        n_e = int(emphasis)
        self.emph = nn.Conv1d(n_e, channels, 3, padding=1) if n_e > 1 else (nn.Conv1d(1, channels, 1) if n_e else None)
        if self.emph is not None:
            nn.init.zeros_(self.emph.weight)
            nn.init.zeros_(self.emph.bias)

    def forward(self, g, emph=None):
        h = self.style(g).unsqueeze(-1)  # [b, C, 1]
        if self.emph is not None and emph is not None:
            h = h + self.emph(emph)  # [b, C, T]
        return h


def load_style_dp(path, model_g, device="cpu"):
    """Load a style SDP into ``model_g.dp`` (in place); returns (DpCond, styles)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    model_g.dp.load_state_dict(ck["dp"])
    w = ck["style"]["weight"] if "weight" in ck["style"] else ck["style"]["style.weight"]
    cond = DpCond(len(ck["styles"]), w.size(1), emphasis=ck.get("emphasis", False)).to(device)
    cond.n_emph = int(ck.get("emphasis", False))
    if "weight" in ck["style"]:  # older checkpoints: bare embedding
        cond.style.load_state_dict(ck["style"])
    else:
        cond.load_state_dict(ck["style"])
    return cond.eval(), list(ck["styles"])


def style_cond(cond, styles, name, device, emph=None):
    """Style name (None = unspecified) [+ emphasis [1, 1, T]] -> SDP h_add."""
    if name and name not in styles:
        raise ValueError(f"unknown style {name!r}; known: {styles}")
    idx = 1 + styles.index(name) if name else 0
    return cond(torch.LongTensor([idx]).to(device), emph)


def _n_channels(emph):
    a = next(iter(emph.values()))
    return 1 if a.ndim == 1 else a.shape[0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--prosody", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--eval-set", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--style-regex", required=True)
    p.add_argument("--style-dropout", type=float, default=0.2)
    p.add_argument("--emphasis", type=Path, help="emphasis.pt: also condition on word emphasis")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--lr", type=float, default=5e-5, help="SDP weights")
    p.add_argument("--style-lr", type=float, default=1e-3, help="style embedding")
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--style-speakers", help="comma-separated styles in speaker-id order (multi-speaker voice)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_g = load_model(args.checkpoint, args.device, False).model_g
    model_g.requires_grad_(False).eval()
    dp = model_g.dp
    assert hasattr(dp, "post_flows"), "needs the stochastic duration predictor"
    dp.requires_grad_(True).train()
    channels = dp.pre.out_channels

    utts = torch.load(args.prosody, weights_only=False)["utts"]
    train = load_rows(args.dataset, utts, "audio")
    val = load_rows(args.eval_set, utts, "utt")
    assert not {u for u, _ in val} & {u for u, _ in train}, "eval utterances in training set"
    rx = re.compile(args.style_regex)
    speakers = args.style_speakers.split(",") if args.style_speakers else None
    names = {u: rx.search(u).group(1) for u, _ in train + val}
    styles = sorted(set(names.values()))
    sidx = {u: 1 + styles.index(n) for u, n in names.items()}
    _LOG.info("train %d, val %d; styles %s", len(train), len(val), styles)

    emph = torch.load(args.emphasis, weights_only=False)["emphasis"] if args.emphasis else None
    n_e = 0 if emph is None else _n_channels(emph)
    emb = DpCond(len(styles), channels, emphasis=n_e if n_e > 1 else n_e > 0).to(args.device)
    opt = torch.optim.AdamW(
        [{"params": dp.parameters(), "lr": args.lr}, {"params": emb.parameters(), "lr": args.style_lr}],
        weight_decay=0.0,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps))
    )

    def batch_loss(rows, p_drop=0.0, labelled=True):
        n = max(len(i) for _, i in rows)
        ids = torch.zeros(len(rows), n, dtype=torch.long)
        w = torch.zeros(len(rows), 1, n)
        for b, (u, pid) in enumerate(rows):
            ids[b, : len(pid)] = torch.LongTensor(pid)
            d = utts[u]["dur"]
            assert len(d) == len(pid), u
            w[b, 0, : len(pid)] = torch.from_numpy(d.astype(np.float32))
        lengths = torch.LongTensor([len(i) for _, i in rows]).to(args.device)
        ids, w = ids.to(args.device), w.to(args.device)
        x, x_mask = text_hidden(model_g, ids, lengths)
        g = torch.LongTensor([
            0 if (not labelled or random.random() < p_drop) else sidx[u] for u, _ in rows
        ]).to(args.device)
        e = None
        if emph is not None:
            e = torch.zeros(len(rows), n_e, n)
            for b, (u, pid) in enumerate(rows):
                if labelled and random.random() >= p_drop / 2:
                    e[b, :, : len(pid)] = torch.from_numpy(emph[u]).reshape(n_e, -1)
            e = e.to(args.device)
        spk = None
        if speakers:  # multi-speaker voice: the true emotion's speaker embedding, never dropped
            sid = torch.LongTensor([speakers.index(names[u]) for u, _ in rows]).to(args.device)
            spk = model_g.emb_g(sid).unsqueeze(-1)
        nll = dp(x, x_mask, w * x_mask, g=spk, h_add=emb(g, e) * x_mask)
        return nll.sum() / x_mask.sum()

    @torch.no_grad()
    def validate(labelled=True):
        dp.eval()
        with torch.random.fork_rng(devices=[args.device] if args.device != "cpu" else []):
            torch.manual_seed(0)
            ls = [batch_loss(val[i : i + 50], labelled=labelled).item() for _ in range(4) for i in range(0, len(val), 50)]
        dp.train()
        return float(np.mean(ls))

    def save(path, extra):
        torch.save({"dp": dp.state_dict(), "style": emb.state_dict(), "styles": styles,
                    "emphasis": n_e if n_e > 1 else emph is not None, **extra}, path)

    v0 = validate()
    _LOG.info("step 0 val %.4f (original SDP)", v0)
    best, step, t0, acc = v0, 0, time.time(), []
    while step < args.steps:
        random.shuffle(train)
        for i in range(0, len(train) - args.batch_size + 1, args.batch_size):
            loss = batch_loss(train[i : i + args.batch_size], args.style_dropout)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(dp.parameters()) + list(emb.parameters()), 1.0)
            opt.step()
            sched.step()
            step += 1
            acc.append(loss.item())
            if step % 100 == 0:
                _LOG.info("step %d loss %.4f %.1f it/s", step, np.mean(acc), 100 / (time.time() - t0))
                acc, t0 = [], time.time()
            if step % args.val_every == 0 or step == args.steps:
                v, vu = validate(), validate(labelled=False)
                tag = ""
                if v < best:
                    best, tag = v, " *"
                    save(args.out_dir / "best.pt", {"step": step, "val": v})
                save(args.out_dir / "last.pt", {"step": step, "val": v})
                _LOG.info("step %d val %.4f (unspecified %.4f)%s", step, v, vu, tag)
            if step >= args.steps:
                break
    _LOG.info("done, best val %.4f (original %.4f)", best, v0)


if __name__ == "__main__":
    main()
