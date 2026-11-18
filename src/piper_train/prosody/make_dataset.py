"""Build a training dataset dir with per-token prosody features inline.

    python -m piper_train.prosody.make_dataset \
        --dataset-dir data/ljspeech/piper \
        --exclude data/ljspeech-medium/eval_set.jsonl \
        --out-dir data/ljspeech/piper_prosody

Audio/spec cache paths keep pointing at the source dataset (nothing is copied).
The prosody stats are added to config.json under "prosody".
"""

import argparse
import json
from pathlib import Path

import torch

from .features import N_CHANNELS, feature_matrix


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--prosody", type=Path, help="default: <dataset-dir>/prosody.pt")
    p.add_argument("--exclude", type=Path, action="append", default=[], help="jsonl with 'utt' ids to hold out")
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()

    pros = torch.load(a.prosody or a.dataset_dir / "prosody.pt", weights_only=False)
    stats = pros["stats"]
    held = {json.loads(l)["utt"] for f in a.exclude for l in open(f, encoding="utf-8") if l.strip()}

    a.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((a.dataset_dir / "config.json").read_text(encoding="utf-8"))
    cfg["prosody"] = {**stats, "channels": N_CHANNELS}
    (a.out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=4), encoding="utf-8")

    n = skipped = 0
    with open(a.out_dir / "dataset.jsonl", "w", encoding="utf-8") as out:
        for line in open(a.dataset_dir / "dataset.jsonl", encoding="utf-8"):
            row = json.loads(line)
            stem = Path(row["audio_path"]).stem
            utt = pros["utts"].get(stem)
            if stem in held or utt is None or len(utt["dur"]) != len(row["phoneme_ids"]):
                skipped += 1
                continue
            row["prosody"] = [[round(float(v), 3) for v in r] for r in feature_matrix(utt, stats)]
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} utterances to {a.out_dir} ({skipped} held out or skipped, {len(held)} in exclude lists)")


if __name__ == "__main__":
    main()
