"""Word / character error rate of synthesized audio via Whisper ASR.

Reads ``manifest.jsonl`` written by ``synthesize.py`` (reference text per wav),
transcribes with a ``transformers`` Whisper model and scores with ``jiwer``
after Whisper's English text normalisation. Defaults to CPU.

    python -m piper_train.eval.score_wer --wav-dir out/baseline
"""

import argparse
import json
from pathlib import Path

import jiwer
import soundfile as sf
import torch
import torchaudio


def transcribe(wavs, model_name: str, device: str, batch_size: int):
    from transformers import pipeline

    asr = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
        chunk_length_s=30,
    )
    out = {}
    for i in range(0, len(wavs), batch_size):
        batch = []
        for w in wavs[i : i + batch_size]:
            wav, sr = sf.read(w, dtype="float32", always_2d=True)
            wav = torch.from_numpy(wav.mean(axis=1))
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            batch.append({"raw": wav.numpy(), "sampling_rate": 16000})
        for w, res in zip(wavs[i : i + batch_size], asr(batch, batch_size=batch_size)):
            out[Path(w).name] = res["text"]
    return out


def compute_errors(references, hypotheses):
    """references/hypotheses: dict name -> text. Returns (summary, per-file)."""
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    norm = EnglishTextNormalizer({})
    names = sorted(references)
    refs = [norm(references[n]) for n in names]
    hyps = [norm(hypotheses[n]) for n in names]
    per_file = {
        n: jiwer.wer(r, h) if r else float(bool(h))
        for n, r, h in zip(names, refs, hyps)
    }
    summary = {
        "wer": jiwer.wer(refs, hyps),
        "cer": jiwer.cer(refs, hyps),
        "n": len(names),
    }
    return summary, per_file


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wav-dir", type=Path, required=True)
    p.add_argument("--model", default="openai/whisper-small.en")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", type=Path, help="default: <wav-dir>/wer.json")
    args = p.parse_args()

    manifest = [
        json.loads(l)
        for l in open(args.wav_dir / "manifest.jsonl", encoding="utf-8")
        if l.strip()
    ]
    refs = {m["wav"]: m["text"] for m in manifest}
    hyps = transcribe([args.wav_dir / n for n in refs], args.model, args.device, args.batch_size)
    summary, per_file = compute_errors(refs, hyps)
    result = {
        **summary,
        "asr_model": args.model,
        "per_file": {n: {"wer": per_file[n], "hyp": hyps[n], "ref": refs[n]} for n in per_file},
    }
    (args.out or args.wav_dir / "wer.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"WER {summary['wer']:.4f}  CER {summary['cer']:.4f}  (n={summary['n']}, {args.model})")


if __name__ == "__main__":
    main()
