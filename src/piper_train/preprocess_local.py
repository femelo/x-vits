"""LJSpeech-format preprocessing without ``piper_phonemize``.

Equivalent to ``piper_train.preprocess`` (same ``cache_norm_audio``: Silero VAD
trim + spectrogram .pt files, same dataset.jsonl layout) but phonemizes with
``espeak-phonemizer`` and reuses an existing voice ``config.json`` verbatim so the
phoneme id map matches the pretrained checkpoint.

    python -m piper_train.preprocess_local \
        --input-dir data/ljspeech/extracted/LJSpeech-1.1 \
        --output-dir data/ljspeech/piper \
        --config data/ljspeech-medium/config.json --max-workers 6
"""

import argparse
import csv
import json
import logging
import shutil
from multiprocessing import Pool
from pathlib import Path

from .eval.phonemize import PiperPhonemizer
from .norm_audio import cache_norm_audio, make_silence_detector

_LOGGER = logging.getLogger("preprocess_local")
_STATE = {}


def _init(config, cache_dir, sample_rate):
    import torch

    torch.set_num_threads(1)
    _STATE["ph"] = PiperPhonemizer(Path(config))
    _STATE["det"] = make_silence_detector()
    _STATE["cache"] = Path(cache_dir)
    _STATE["sr"] = sample_rate


def _work(item):
    wav, text = item
    try:
        ph = _STATE["ph"]
        phonemes = ph.phonemes(text)
        ids = ph.ids(text)
        norm, spec = cache_norm_audio(wav, _STATE["cache"], _STATE["det"], _STATE["sr"])
        return {
            "text": text,
            "audio_path": str(wav),
            "speaker": None,
            "speaker_id": None,
            "phonemes": phonemes,
            "phoneme_ids": ids,
            "audio_norm_path": str(norm),
            "audio_spec_path": str(spec),
        }
    except Exception:
        _LOGGER.exception("failed: %s", wav)
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--max-workers", type=int, default=6)
    p.add_argument("--limit", type=int, help="only first N utterances (testing)")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    sr = cfg["audio"]["sample_rate"]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    cache = a.output_dir / "cache" / str(sr)
    cache.mkdir(parents=True, exist_ok=True)
    shutil.copy(a.config, a.output_dir / "config.json")

    wav_dir = a.input_dir / ("wavs" if (a.input_dir / "wavs").is_dir() else "wav")
    items = []
    with open(a.input_dir / "metadata.csv", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="|"):
            wav = wav_dir / f"{row[0]}.wav"
            if wav.exists() and wav.stat().st_size > 0:
                items.append((wav, row[-1]))
    if a.limit:
        items = items[: a.limit]
    _LOGGER.info("%d utterances, %d workers", len(items), a.max_workers)

    n = bad = 0
    with Pool(a.max_workers, _init, (a.config, cache, sr)) as pool, open(
        a.output_dir / "dataset.jsonl", "w", encoding="utf-8"
    ) as out:
        for r in pool.imap(_work, items, chunksize=16):
            if r is None:
                bad += 1
                continue
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
            if n % 1000 == 0:
                _LOGGER.info("%d done", n)
    _LOGGER.info("wrote %d utterances (%d failed)", n, bad)


if __name__ == "__main__":
    main()
