"""Stationary-noise reduction by spectral gating, for cleaning TTS training recordings.

For each file, the noise spectrum is estimated from its quietest frames (pauses,
breath gaps); every STFT bin is then attenuated by a Wiener-like gain
``max(1 - over * noise / power, floor)``, smoothed over time and frequency so it
doesn't produce musical noise.  The reduction is capped (``max_db``), so speech
well above the noise, whispered speech included, passes nearly unchanged while
the background between words drops by up to ``max_db``.  Full band, no model.

    python -m piper_train.norm_audio.spectral_gate --input-dir in/ --output-dir out/ [--max-db 15]
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter
from scipy.signal import istft, stft

N_FFT, HOP = 1024, 256


def spectral_gate(x: np.ndarray, sr: int, max_db: float = 15.0, noise_pct: float = 10.0,
                  over: float = 1.5, smooth=(3, 5)) -> np.ndarray:
    """x: mono float audio -> denoised audio of the same length."""
    _, _, Z = stft(x, sr, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary="even")
    P = np.abs(Z) ** 2                                   # [bins, frames]
    e = P.sum(0)
    live = e > e.max() * 1e-9          # ignore digital silence (e.g. padding): it isn't the room noise
    ref = e[live] if live.any() else e
    quiet = live & (e <= np.percentile(ref, noise_pct))
    noise = P[:, quiet].mean(1, keepdims=True) if quiet.any() else P.min(1, keepdims=True)
    floor = 10 ** (-max_db / 20)
    gain = np.clip(1.0 - over * noise / np.maximum(P, 1e-20), floor ** 2, 1.0) ** 0.5   # amplitude gain
    gain = uniform_filter(gain, size=smooth, mode="nearest")                            # freq x time smoothing
    _, y = istft(Z * gain, sr, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=True)
    return y[: len(x)].astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-db", type=float, default=15.0, help="largest attenuation of the background (dB)")
    p.add_argument("--only", help="comma-separated subfolders (styles) to process; others are copied unchanged")
    p.add_argument("--style-max-db", default="",
                   help='per-style cap for flat folders with <style>_<id>.wav names, e.g. "whisper=25,sleepy=25"')
    a = p.parse_args()
    only = set(a.only.split(",")) if a.only else None
    per_style = {k: float(v) for k, v in (kv.split("=") for kv in a.style_max_db.split(",") if kv)}
    files = sorted(a.input_dir.rglob("*.wav"))
    for f in files:
        rel = f.relative_to(a.input_dir)
        x, sr = sf.read(f, dtype="float32")
        style = rel.parts[0] if len(rel.parts) > 1 else f.stem.split("_", 1)[0]
        if only is None or style in only:
            x = spectral_gate(x, sr, per_style.get(style, a.max_db))
        (a.output_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        sf.write(a.output_dir / rel, x, sr, subtype="PCM_16")
    print(f"{len(files)} files -> {a.output_dir}")


if __name__ == "__main__":
    main()
