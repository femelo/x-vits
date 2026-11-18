#!/usr/bin/env bash
# The voice behind the README numbers: Expresso speaker ex02 (CC BY-NC 4.0), read styles only.
#
# Prepare data/expresso_ex02/ in LJSpeech layout:
#   wavs/ex02_<style>_<n>.wav  (22.05 kHz mono)
#   metadata.csv               id|text, with Expresso's *emphasis* marks kept
# (whisper, singing, longform and <laugh> rows were left out: 2.2 h remain)
set -euo pipefail
xvits-train \
  --dataset data/expresso_ex02 \
  --base-ckpt checkpoints/lj-med_1000.ckpt \
  --base-config checkpoints/config.json \
  --out runs/expresso_ex02 \
  --style-regex '^ex\d+_([a-z]+)_' \
  --ft-epochs 400 --voice-epochs 200 \
  --compile --export
