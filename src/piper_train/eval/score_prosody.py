"""Prosody agreement between synthesized audio and the real recording of the same text.

Both sides are aligned with the same pretrained VITS aligner and pitch-tracked the
same way (``piper_train.prosody.extract``), so this works for any system that
produces wavs.  Tokens are grouped into units (phoneme + its trailing pad).

    python -m piper_train.eval.score_prosody --wav-dir out/baseline \
        --eval-set data/ljspeech-medium/eval_set.jsonl \
        --prosody data/ljspeech/piper/prosody.pt \
        --checkpoint data/ljspeech-medium/lj-med_1000.ckpt

Reports (mean over utterances): duration correlation / error, pause agreement,
pitch-contour correlation / error in semitones, pitch-range ratio, energy correlation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from ..prosody import extract as ex

PAUSE_SYMBOLS = " ,.;:!?—…\"()"


def units(feats, hop_s):
    """Group tokens pairwise: (BOS,pad), (p1,pad), ..., (EOS)."""
    dur, lf0, vf, en = (feats[k].astype(np.float64) for k in ("dur", "lf0", "vfrac", "energy"))
    n = (len(dur) + 1) // 2
    out = {"dur": np.zeros(n), "lf0": np.full(n, np.nan), "energy": np.zeros(n)}
    for i in range(n):
        sl = slice(2 * i, 2 * i + 2)
        d = dur[sl]
        out["dur"][i] = d.sum() * hop_s
        w = d * vf[sl]
        ok = ~np.isnan(lf0[sl]) & (w > 0)
        if ok.any() and w[ok].sum() >= 0.5 * d.sum():  # mostly voiced
            out["lf0"][i] = (lf0[sl][ok] * w[ok]).sum() / w[ok].sum()
        out["energy"][i] = (en[sl] * np.maximum(d, 1e-3)).sum() / np.maximum(d, 1e-3).sum()
    return out


def semitones(lf0):
    return 12.0 / np.log(2.0) * (lf0 - np.nanmedian(lf0))


def corr(a, b):
    if len(a) < 3 or a.std() < 1e-8 or b.std() < 1e-8:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def compare(ref, syn, ids, pause_ids, hop_s):
    r, s = units(ref, hop_s), units(syn, hop_s)
    unit_ids = np.array(ids[::2])
    inner = np.ones(len(unit_ids), bool)
    inner[[0, -1]] = False  # BOS / EOS absorb leading / trailing silence
    is_pause = np.isin(unit_ids, pause_ids) & inner
    speech = inner & ~is_pause
    m = {}
    m["dur_corr"] = corr(np.log(r["dur"][speech] + 0.005), np.log(s["dur"][speech] + 0.005))
    m["dur_mae_ms"] = float(np.abs(r["dur"][speech] - s["dur"][speech]).mean() * 1000)
    m["rate_ratio"] = float(r["dur"][inner].sum() / s["dur"][inner].sum())  # >1: synth faster
    if is_pause.sum() >= 3:
        m["pause_corr"] = corr(r["dur"][is_pause], s["dur"][is_pause])
    m["pause_ratio"] = float(s["dur"][is_pause].sum() / max(r["dur"][is_pause].sum(), 1e-6))
    rs, ss = semitones(r["lf0"]), semitones(s["lf0"])
    both = speech & ~np.isnan(rs) & ~np.isnan(ss)
    m["f0_corr"] = corr(rs[both], ss[both])
    m["f0_rmse_st"] = float(np.sqrt(((rs[both] - ss[both]) ** 2).mean())) if both.sum() else np.nan
    fr, fs = (f["f0"].astype(np.float64) for f in (ref, syn))
    sd = lambda f: (12 * np.log2(f[f > 0] / np.median(f[f > 0]))).std()
    m["f0_std_ratio"] = float(sd(fs) / sd(fr))  # <1: synth flatter
    m["energy_corr"] = corr(r["energy"][speech], s["energy"][speech])
    return m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wav-dir", type=Path, required=True)
    p.add_argument("--eval-set", type=Path, required=True, help="jsonl with id, utt, phoneme_ids")
    p.add_argument("--prosody", type=Path, required=True, help="reference prosody.pt")
    p.add_argument("--checkpoint", type=Path, required=True, help="aligner checkpoint")
    p.add_argument("--config", type=Path, help="default: <eval-set dir>/config.json")
    p.add_argument("--out", type=Path, help="default: <wav-dir>/prosody.json")
    a = p.parse_args()
    torch.set_num_threads(4)

    ref = torch.load(a.prosody, weights_only=False)
    st = ref["stats"]
    ex._STATE["fmin"], ex._STATE["fmax"] = st["f0_range"]
    hop_s = st["hop"] / st["sample_rate"]
    cfg = json.loads((a.config or a.eval_set.parent / "config.json").read_text(encoding="utf-8"))
    pause_ids = [v[0] for k, v in cfg["phoneme_id_map"].items() if k in PAUSE_SYMBOLS]
    model = ex.load_aligner(a.checkpoint)

    per_file = {}
    for line in open(a.eval_set, encoding="utf-8"):
        e = json.loads(line)
        wav = a.wav_dir / f"{e['id']}.wav"
        if not wav.exists() or e.get("utt") not in ref["utts"]:
            continue
        r = ref["utts"][e["utt"]]
        if len(r["dur"]) != len(e["phoneme_ids"]):
            continue
        audio, sr = sf.read(wav, dtype="float32", always_2d=True)
        assert sr == st["sample_rate"], sr
        syn = ex.extract(model, e["phoneme_ids"], torch.from_numpy(audio.mean(1)), sr)
        per_file[wav.name] = compare(r, syn, e["phoneme_ids"], pause_ids, hop_s)

    keys = sorted({k for m in per_file.values() for k in m})
    summary = {k: float(np.nanmean([m.get(k, np.nan) for m in per_file.values()])) for k in keys}
    summary["n"] = len(per_file)
    out = a.out or a.wav_dir / "prosody.json"
    out.write_text(json.dumps({**summary, "per_file": per_file}, indent=1))
    print("  ".join(f"{k} {v:.3f}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
