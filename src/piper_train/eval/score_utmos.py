"""Score synthesized audio with UTMOS22 (strong).

Uses a local checkout of ``tarepan/SpeechMOS`` (third_party/speechmos_repo,
symlinked to ~/pocket-tts-distill/extra_pkgs/speechmos_repo) with its local
weights; nothing is downloaded.
Defaults to CPU.

    python -m piper_train.eval.score_utmos --wav-dir out/baseline
"""

import argparse
import json
import statistics
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

import os

# Local SpeechMOS checkout + weights (no network, no remote code). Override with
# SPEECHMOS_DIR / SPEECHMOS_WEIGHTS.
_DEFAULT_DIR = Path(__file__).resolve().parents[5] / "third_party" / "speechmos_repo"
SPEECHMOS_DIR = Path(os.environ.get("SPEECHMOS_DIR", _DEFAULT_DIR))
SPEECHMOS_WEIGHTS = Path(
    os.environ.get("SPEECHMOS_WEIGHTS", SPEECHMOS_DIR / "utmos22_strong.pt")
)


def load_predictor(device: str):
    model = torch.hub.load(
        str(SPEECHMOS_DIR), "utmos22_strong", source="local", pretrained=False
    )
    model.load_state_dict(torch.load(SPEECHMOS_WEIGHTS, map_location="cpu"))
    return model.eval().to(device)


def score_files(wavs, device: str = "cpu"):
    predictor = load_predictor(device)
    scores = {}
    for wav_path in wavs:
        wav, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav.mean(axis=1))  # mono
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
            sr = 16000
        with torch.no_grad():
            scores[Path(wav_path).name] = float(
                predictor(wav.unsqueeze(0).to(device), sr).item()
            )
    return scores


def summarize(scores):
    vals = list(scores.values())
    return {
        "utmos_mean": statistics.fmean(vals),
        "utmos_std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "n": len(vals),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wav-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, help="default: <wav-dir>/utmos.json")
    args = p.parse_args()

    wavs = sorted(args.wav_dir.glob("*.wav"))
    assert wavs, f"no wav files in {args.wav_dir}"
    scores = score_files(wavs, args.device)
    result = {**summarize(scores), "per_file": scores}
    (args.out or args.wav_dir / "utmos.json").write_text(json.dumps(result, indent=2))
    print(f"UTMOS {result['utmos_mean']:.3f} ± {result['utmos_std']:.3f} (n={result['n']})")


if __name__ == "__main__":
    main()
