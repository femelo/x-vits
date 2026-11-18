"""Stage 1: per-token prosody targets (duration, pitch, energy) for a dataset.

Uses the pretrained VITS model itself as the aligner (posterior encoder + flow +
monotonic alignment search), so durations are on the checkpoint's own token grid
(Piper layout: BOS, pad, p1, pad, ..., pn, pad, EOS).  Pitch comes from
``librosa.pyin`` on the same frame grid as the spectrogram.

    NUMBA_CPU_NAME=generic python -m piper_train.prosody.extract \
        --dataset-dir data/ljspeech/piper \
        --checkpoint data/ljspeech-medium/lj-med_1000.ckpt --max-workers 6

Writes ``<dataset-dir>/prosody.pt``: {"utts": {stem: {...}}, "stats": {...}}.
Per utterance: dur [n_tok] int16 (frames), lf0 [n_tok] float32 (mean log-Hz over
voiced frames, NaN if none), vfrac [n_tok], energy [n_tok] (mean log spectral
norm), f0 [n_frames] float16 (Hz, 0 = unvoiced).
"""

import argparse
import json
import logging
import os
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("NUMBA_CPU_NAME", "generic")  # pyin segfaults otherwise here

import numpy as np
import torch

_LOGGER = logging.getLogger("prosody.extract")
HOP, WIN = 256, 1024
FMIN, FMAX = 70.0, 500.0
_STATE = {}


def load_aligner(checkpoint):
    from ..eval.synthesize import load_model

    return load_model(Path(checkpoint), "cpu", use_cfm=False).model_g


def frame_f0(audio: np.ndarray, sr: int, n_frames: int) -> np.ndarray:
    """F0 in Hz per spectrogram frame (0 = unvoiced).

    The spectrogram (center=False, reflect pad (WIN-HOP)/2) has frame i centred at
    sample i*HOP + HOP/2; pyin(center=True) centres frame i at i*HOP, so shift.
    """
    import librosa

    f0, voiced, _ = librosa.pyin(
        audio[HOP // 2 :],
        fmin=_STATE.get("fmin", FMIN),
        fmax=_STATE.get("fmax", FMAX),
        sr=sr,
        frame_length=WIN,
        hop_length=HOP,
    )
    f0 = np.where(voiced & ~np.isnan(f0), f0, 0.0).astype(np.float32)
    if len(f0) < n_frames:
        f0 = np.pad(f0, (0, n_frames - len(f0)))
    return f0[:n_frames]


@torch.no_grad()
def align(model, phoneme_ids, spec):
    """Token durations (frames) from MAS under the pretrained flow prior."""
    from ..vits import monotonic_align

    x = torch.LongTensor(phoneme_ids).unsqueeze(0)
    x_len = torch.LongTensor([x.size(1)])
    y = spec.unsqueeze(0)
    y_len = torch.LongTensor([y.size(2)])
    _, m_p, logs_p, x_mask = model.enc_p(x, x_len)
    z, _, _, y_mask = model.enc_q(y, y_len, g=None)
    z_p = model.flow(z, y_mask, g=None)
    attn_mask = x_mask.unsqueeze(2) * y_mask.unsqueeze(-1)
    neg_cent = model._neg_cent_gaussian(z_p, m_p, logs_p)
    attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1))  # [1, t_y, t_x]
    return attn[0].sum(0).long().numpy() if attn.shape[1] == y.size(2) else attn[0].sum(1).long().numpy()


def token_features(dur, f0, spec):
    """Aggregate frame-level f0 / energy to tokens."""
    energy_f = torch.log(spec.norm(dim=0).clamp_min(1e-5)).numpy()
    n = len(dur)
    lf0 = np.full(n, np.nan, np.float32)
    vfrac = np.zeros(n, np.float32)
    energy = np.zeros(n, np.float32)
    ends = np.cumsum(dur)
    for i, (s, e) in enumerate(zip(ends - dur, ends)):
        if e <= s:
            energy[i] = energy_f[min(s, len(energy_f) - 1)]
            continue
        seg = f0[s:e]
        v = seg > 0
        vfrac[i] = v.mean()
        if v.any():
            lf0[i] = np.log(seg[v]).mean()
        energy[i] = energy_f[s:e].mean()
    return lf0, vfrac, energy


def extract(model, phoneme_ids, audio: torch.Tensor, sr: int, spec=None):
    """audio: [T] float in [-1, 1].  Returns the per-utterance feature dict."""
    if spec is None:
        from ..vits.mel_processing import spectrogram_torch

        spec = spectrogram_torch(
            y=audio.unsqueeze(0), n_fft=WIN, sampling_rate=sr, hop_size=HOP, win_size=WIN, center=False
        ).squeeze(0)
    dur = align(model, phoneme_ids, spec)
    assert dur.sum() == spec.size(1) and len(dur) == len(phoneme_ids), (dur.sum(), spec.size(1), len(dur))
    f0 = frame_f0(audio.numpy(), sr, spec.size(1))
    lf0, vfrac, energy = token_features(dur, f0, spec)
    return {
        "dur": dur.astype(np.int16),
        "lf0": lf0,
        "vfrac": vfrac,
        "energy": energy,
        "f0": f0.astype(np.float16),
    }


def speaker_f0_range(rows, sr, n=40, seed=0):
    """Narrow the pitch search to the speaker (cuts octave errors): one pass with
    the wide range on a sample, then [median / 2.2, median * 1.9]."""
    rng = np.random.default_rng(seed)
    med = []
    for i in rng.choice(len(rows), size=min(n, len(rows)), replace=False):
        audio = torch.load(rows[i]["audio_norm_path"]).squeeze(0).numpy()
        f0 = frame_f0(audio, sr, len(audio) // HOP)
        if (f0 > 0).any():
            med.append(np.median(f0[f0 > 0]))
    m = float(np.median(med))
    return m / 2.2, m * 1.9


def _init(checkpoint, sr, f0_range=None):
    torch.set_num_threads(1)
    if f0_range:
        _STATE["fmin"], _STATE["fmax"] = f0_range
    _STATE["model"] = load_aligner(checkpoint)
    _STATE["sr"] = sr


def _work(row):
    try:
        audio = torch.load(row["audio_norm_path"]).squeeze(0)
        spec = torch.load(row["audio_spec_path"])
        feats = extract(_STATE["model"], row["phoneme_ids"], audio, _STATE["sr"], spec=spec)
        return Path(row["audio_path"]).stem, feats
    except Exception:
        _LOGGER.exception("failed: %s", row.get("audio_path"))
        return Path(row["audio_path"]).stem, None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--max-workers", type=int, default=6)
    p.add_argument("--limit", type=int)
    p.add_argument("--out", type=Path, help="default: <dataset-dir>/prosody.pt")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    sr = json.loads((a.dataset_dir / "config.json").read_text())["audio"]["sample_rate"]
    rows = [json.loads(l) for l in open(a.dataset_dir / "dataset.jsonl", encoding="utf-8")]
    if a.limit:
        rows = rows[: a.limit]
    f0_range = speaker_f0_range(rows, sr)
    _LOGGER.info("speaker pitch search range: %.0f-%.0f Hz", *f0_range)
    utts, bad = {}, 0
    with Pool(a.max_workers, _init, (a.checkpoint, sr, f0_range)) as pool:
        for i, (stem, feats) in enumerate(pool.imap_unordered(_work, rows, chunksize=8), 1):
            if feats is None:
                bad += 1
            else:
                utts[stem] = feats
            if i % 1000 == 0:
                _LOGGER.info("%d / %d", i, len(rows))

    lf0 = np.concatenate([u["lf0"][~np.isnan(u["lf0"])] for u in utts.values()])
    en = np.concatenate([u["energy"] for u in utts.values()])
    stats = {
        "lf0_mean": float(lf0.mean()),
        "lf0_std": float(lf0.std()),
        "energy_mean": float(en.mean()),
        "energy_std": float(en.std()),
        "f0_range": list(f0_range),
        "sample_rate": sr,
        "hop": HOP,
        "n_utts": len(utts),
    }
    out = a.out or a.dataset_dir / "prosody.pt"
    torch.save({"utts": utts, "stats": stats}, out)
    _LOGGER.info("wrote %s: %d utterances (%d failed); stats %s", out, len(utts), bad, stats)


if __name__ == "__main__":
    main()
