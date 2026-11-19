#!/usr/bin/env bash
# German: Thorsten-Voice Piper-Scratch + Thorsten's emotional dataset (OpenSLR 110, CC0).
#
# Base checkpoint (Hugging Face Thorsten-Voice/Piper-Scratch):
#   checkpoints/epoch=1799-last.ckpt and de_DE-thorsten_scratch-medium.onnx.json
#
# Prepare data/thorsten_emotional/ in LJSpeech layout:
#   wavs/<emotion>_<guid>.wav   (22.05 kHz mono, RMS-normalised, ~150 ms silence padded at both ends:
#                                the original files are cut hard at the speech boundaries)
#   metadata.csv                <emotion>_<guid>|text  (the same 300 sentences in all 8 emotions)
#   dev_ids.txt                 every emotion of 15 held-out sentences (120 ids), so no dev text
#                               is seen in training
set -euo pipefail
xvits-train \
  --dataset data/thorsten_emotional \
  --base-ckpt "checkpoints/thorsten/epoch=1799-last.ckpt" \
  --base-config checkpoints/thorsten/de_DE-thorsten_scratch-medium.onnx.json \
  --dev-list data/thorsten_emotional/dev_ids.txt \
  --style-regex '^([a-z]+)_' \
  --out runs/thorsten_emotional \
  --compile --export
