"""Download the reference-voice assets used by the test/eval harness.

Default voice: en_US-ljspeech-medium (lessac wavs need Blizzard credentials, amy's are unavailable). Everything comes
from the ``rhasspy/piper-checkpoints`` dataset repo:

* ``config.json`` and ``dataset.jsonl.gz`` (text + phoneme ids; always fetched)
* the ~845 MB training ``.ckpt`` (only with ``--ckpt``)

It also writes a deterministic held-out ``eval_set.jsonl`` (text + phoneme ids),
so evaluation needs neither the original wavs nor espeak/piper-phonemize.
"""

import argparse
import gzip
import json
import random
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "rhasspy/piper-checkpoints"
VOICES = {
    "lessac": ("en/en_US/lessac/medium", "epoch=2164-step=1355540.ckpt"),
    "amy": ("en/en_US/amy/medium", "epoch=6679-step=1554200.ckpt"),
    # trained from scratch on LJSpeech (public domain); no dataset.jsonl.gz upstream,
    # build the eval set from LJSpeech's metadata.csv with --ljspeech-metadata
    "ljspeech": ("en/en_US/ljspeech/medium", "lj-med_1000.ckpt"),
}


def fetch(subdir: str, name: str, out_dir: Path) -> Path:
    cached = hf_hub_download(REPO, f"{subdir}/{name}", repo_type="dataset")
    dest = out_dir / name
    shutil.copyfile(cached, dest)
    return dest


def build_eval_set_from_ljspeech(metadata_csv: Path, config: Path, out_path: Path, n: int, seed: int) -> None:
    """LJSpeech metadata.csv is ``id|raw text|normalized text``; phonemize the latter."""
    from .phonemize import PiperPhonemizer

    ph = PiperPhonemizer(config)
    rows = []
    for line in open(metadata_csv, encoding="utf-8"):
        parts = line.rstrip("\n").split("|")
        if len(parts) >= 3 and 20 <= len(parts[2]) <= 200:
            rows.append((parts[0], parts[2]))
    random.Random(seed).shuffle(rows)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (utt, text) in enumerate(rows[:n]):
            row = {"id": f"{i:04d}", "utt": utt, "text": text, "phoneme_ids": ph.ids(text)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_eval_set(dataset_gz: Path, out_path: Path, n: int, seed: int) -> None:
    with gzip.open(dataset_gz, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = [r for r in rows if r.get("text") and 20 <= len(r["text"]) <= 200]
    rng = random.Random(seed)
    rng.shuffle(rows)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows[:n]):
            f.write(
                json.dumps(
                    {"id": f"{i:04d}", "text": r["text"], "phoneme_ids": r["phoneme_ids"]},
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_eval_set_from_text(text_file: Path, config: Path, out_path: Path) -> None:
    """Phonemize one sentence per line (unseen text -> fairer generalisation check)."""
    from .phonemize import PiperPhonemizer

    ph = PiperPhonemizer(config)
    lines = [l.strip() for l in open(text_file, encoding="utf-8") if l.strip()]
    with open(out_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(lines):
            row = {"id": f"{i:04d}", "text": text, "phoneme_ids": ph.ids(text)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--voice", choices=sorted(VOICES), default="ljspeech")
    p.add_argument("--ljspeech-metadata", type=Path, help="LJSpeech-1.1/metadata.csv")
    p.add_argument("--out-dir", type=Path, help="default: data/<voice>-medium")
    p.add_argument("--ckpt", action="store_true", help="also download the 845 MB .ckpt")
    p.add_argument("--num-sentences", type=int, default=100)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--text-file",
        type=Path,
        help="build the eval set from new sentences (one per line) via espeak "
        "instead of sampling amy's training manifest",
    )
    args = p.parse_args()

    subdir, ckpt = VOICES[args.voice]
    args.out_dir = args.out_dir or Path(f"data/{args.voice}-medium")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fetch(subdir, "config.json", args.out_dir)
    has_manifest = args.voice != "ljspeech"
    ds = fetch(subdir, "dataset.jsonl.gz", args.out_dir) if has_manifest else None
    if args.ckpt:
        fetch(subdir, ckpt, args.out_dir)
    out = args.out_dir / "eval_set.jsonl"
    if args.ljspeech_metadata:
        build_eval_set_from_ljspeech(
            args.ljspeech_metadata, args.out_dir / "config.json", out, args.num_sentences, args.seed
        )
    elif args.text_file:
        build_eval_set_from_text(args.text_file, args.out_dir / "config.json", out)
    elif ds is not None:
        build_eval_set(ds, out, args.num_sentences, args.seed)
    print(f"Wrote assets to {args.out_dir}")


if __name__ == "__main__":
    main()
