"""Score synthesized audio with UTMOS22 (strong).

Uses a local checkout of ``tarepan/SpeechMOS`` when ``SPEECHMOS_DIR`` points to one
(weights at ``$SPEECHMOS_DIR/utmos22_strong.pt`` or ``SPEECHMOS_WEIGHTS``); otherwise
loads ``tarepan/SpeechMOS`` from torch hub, which downloads its code and weights once.
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

SPEECHMOS_HUB = "tarepan/SpeechMOS:v1.2.0"


def load_predictor(device: str):
    local = os.environ.get("SPEECHMOS_DIR")
    if local:  # local checkout + weights: no network, no remote code
        weights = Path(os.environ.get("SPEECHMOS_WEIGHTS", Path(local) / "utmos22_strong.pt"))
        model = torch.hub.load(local, "utmos22_strong", source="local", pretrained=False)
        model.load_state_dict(torch.load(weights, map_location="cpu"))
    else:
        model = torch.hub.load(SPEECHMOS_HUB, "utmos22_strong", trust_repo=True)
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
