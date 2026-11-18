"""Fit the stress-anchored emphasis template (stage 4a/4b) on training recordings.

For every emphasised word, pitch and log duration are measured per token as a
function of the signed token offset from the word's stressed vowel, relative to
the same measurements on non-emphasised words of the same sentences.  A small
parametric shape is fitted to the difference (see ``emphasis.shape_pitch`` /
``emphasis.shape_logdur``): one pooled shape, plus per-style strengths when a
style regex is given and the style has enough words.

Which words count as "emphasised":
  * with ``--marked`` files (``id|...|text with *marks*``): the marked words;
  * otherwise: words whose acoustic prominence (``prominence.word_prominence``)
    is >= ``--z-thr``, which works for any voice without annotation.

    python -m piper_train.prosody.fit_template \\
        --prosody prosody_all.pt --dataset piper_s2/dataset.jsonl --config config.json \\
        [--marked metadata_marked.csv] [--style-regex '^ex\\d+_([a-z]+)_'] \\
        --out emphasis_template.json
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares

from ..eval.phonemize import PiperPhonemizer
from .emphasis import parse_marked, shape_logdur, shape_pitch, token_word_index, tonic_offsets
from .prominence import vowel_duration_means, word_prominence

K = np.arange(-8, 15)
PITCH_NAMES = ["A", "mu", "s_l", "s_r", "R", "k_r", "tau"]
DUR_NAMES = ["C", "B", "mu", "s_l", "s_r"]
# smooth release (tau >= 1.5 tokens); duration rise only a few tokens wide
PITCH_X0, PITCH_LO, PITCH_HI = [2.5, 1, 1.5, 2.5, 1.5, 5, 2], [0, -3, 0.3, 0.3, 0, 0, 1.5], [8, 4, 6, 8, 5, 12, 4]
DUR_X0, DUR_LO, DUR_HI = [0.4, 0.2, -1, 1, 3], [-0.5, -1, -4, 0.3, 0.3], [1.5, 1.5, 4, 2.5, 10]


def _fit(y, n, fn, names, x0, lo, hi, min_count=15):
    ok = ~np.isnan(y) & (n >= min_count)
    if ok.sum() < len(names):
        return None, float("nan")
    w = np.sqrt(n[ok])
    r = least_squares(lambda q: (fn(K[ok], dict(zip(names, q))) - y[ok]) * w, x0, bounds=(lo, hi))
    p = dict(zip(names, (float(v) for v in r.x)))
    rmse = float(np.sqrt(np.mean((fn(K[ok], p) - y[ok]) ** 2)))
    return {k: round(v, 3) for k, v in p.items()}, rmse


def collect(utts, rows, pz, marked=None, style_of=None, z_thr=1.0):
    """rows: [(utt, row)] -> {group: {emph?: {k: ([pitch], [logdur])}}}"""
    inv = {v: k for k, v in pz.id_map.items()}
    vmeans = vowel_duration_means(utts, [(u, r["phoneme_ids"]) for u, r in rows], inv)
    data = {}
    for u, r in rows:
        if marked is not None:
            t = marked.get(u)
            if not t or "*" not in t:
                continue
            plain, flags = parse_marked(t)
            ids, tw = token_word_index(plain, pz)
            if ids != r["phoneme_ids"]:
                continue
        else:
            ids, tw = token_word_index(r["text"], pz)
            if ids != r["phoneme_ids"]:
                continue
            sc = word_prominence(utts[u], ids, tw, inv, vmeans)
            flags = [bool(s >= z_thr) for s in np.nan_to_num(sc, nan=-9)]
            if not any(flags):
                continue
        x = utts[u]
        st = 12 * np.nan_to_num(x["lf0"]) / np.log(2)
        v = x["vfrac"] > 0.5
        d = np.log(x["dur"].astype(float) + 0.5)
        twa = np.array(tw)
        valid = twa >= 0
        if not (v & valid).any():
            continue
        med, dm = np.median(st[v & valid]), d[valid].mean()
        off = tonic_offsets(ids, tw, inv)
        groups = ["all"] + ([style_of(u)] if style_of else [])
        for wi, f in enumerate(flags):
            for p in np.nonzero(twa == wi)[0]:
                k = int(off[p]) if not np.isnan(off[p]) else None
                if k is None or k < K[0] or k > K[-1]:
                    continue
                for g in groups:
                    e = data.setdefault(g, {m: {kk: ([], []) for kk in K} for m in (True, False)})
                    if v[p]:
                        e[bool(f)][k][0].append(st[p] - med)
                    e[bool(f)][k][1].append(d[p] - dm)
    return data


def profiles(e):
    mean = lambda a, fill: np.array([np.mean(a[k]) if len(a[k]) else fill for k in K])
    mp = mean({k: e[True][k][0] for k in K}, np.nan)
    up = mean({k: e[False][k][0] for k in K}, 0.0)
    md = mean({k: e[True][k][1] for k in K}, np.nan)
    ud = mean({k: e[False][k][1] for k in K}, 0.0)
    n = np.array([len(e[True][k][1]) for k in K])
    return mp - up, md - ud, n


def fit_all(data, min_style_words=40):
    if "all" not in data:
        raise SystemExit("no emphasised words found: nothing to fit")
    dp, dd, n = profiles(data["all"])
    pp, pe = _fit(dp, n, shape_pitch, PITCH_NAMES, PITCH_X0, PITCH_LO, PITCH_HI)
    pd, de = _fit(dd, n, shape_logdur, DUR_NAMES, DUR_X0, DUR_LO, DUR_HI)
    if pp is None or pd is None:
        raise SystemExit("too few emphasised words to fit a template")
    out = {"anchor": "stressed vowel, token offsets", "fit_range": [int(K[0]), int(K[-1])],
           "pitch": pp, "dur": pd, "n_words": int(n[K == 0][0]),
           "rmse_pitch_st": round(pe, 2), "rmse_logdur": round(de, 3), "styles": {}}
    for g in sorted(k for k in data if k != "all"):
        dp, dd, n = profiles(data[g])
        nw = int(n[K == 0][0])
        if nw < min_style_words:
            continue
        # per style: strengths only; positions and widths come from the pooled fit
        fp = lambda k, q: shape_pitch(k, dict(pp, A=q["A"], R=q["R"]))
        fd = lambda k, q: shape_logdur(k, dict(pd, C=q["C"], B=q["B"]))
        a, pe_s = _fit(dp, n, fp, ["A", "R"], [pp["A"], pp["R"]], [0, 0], [10, 5])
        b, de_s = _fit(dd, n, fd, ["C", "B"], [pd["C"], pd["B"]], [-0.5, -1], [1.5, 1.5])
        if a is None or b is None:
            continue
        entry = {"pitch": dict(pp, **a), "dur": dict(pd, **b), "n_words": nw,
                 "rmse_pitch_st": round(pe_s, 2), "rmse_logdur": round(de_s, 3)}
        if entry["pitch"]["A"] < 0.5 * pp["A"]:  # the shared shape does not fit this style's pitch
            entry["pitch"] = dict(pp)
            entry["pitch_from"] = "pooled (style shape does not fit)"
        out["styles"][g] = entry
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prosody", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True, help="training dataset.jsonl")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--marked", type=Path, action="append", default=[])
    p.add_argument("--style-regex")
    p.add_argument("--z-thr", type=float, default=1.0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    pz = PiperPhonemizer(a.config)
    utts = torch.load(a.prosody, weights_only=False)["utts"]
    rows = [(Path(r["audio_path"]).stem, r) for r in map(json.loads, open(a.dataset, encoding="utf-8")) if r]
    rows = [(u, r) for u, r in rows if u in utts]
    marked = None
    if a.marked:
        marked = {}
        for f in a.marked:
            for line in open(f, encoding="utf-8"):
                q = line.rstrip("\n").split("|")
                marked[q[0]] = q[-1]
    style_of = None
    if a.style_regex:
        rx = re.compile(a.style_regex)
        style_of = lambda u: (rx.search(u).group(1) if rx.search(u) else "default")
    out = fit_all(collect(utts, rows, pz, marked, style_of, a.z_thr))
    out["source"] = "human marks" if marked is not None else f"acoustic prominence (z >= {a.z_thr})"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{out['source']}: {out['n_words']} words; pitch peak {out['pitch']['A']:+.2f} st, "
          f"stressed-syllable lengthening x{np.exp(out['dur']['B']):.2f}; styles: {list(out['styles'])}")


if __name__ == "__main__":
    main()
