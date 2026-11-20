#!/usr/bin/env bash
# German: Thorsten's emotional dataset (OpenSLR 110, CC0) on Piper's thorsten/high voice (CC0),
# with an emotion embedding in the voice, cleaned recordings and per-style output levels.
#
# Base checkpoint (Hugging Face dataset rhasspy/piper-checkpoints, de/de_DE/thorsten/high):
#   epoch=2665-step=1182078.ckpt and config.json
#
# 1. Prepare data/thorsten_emotional/ in LJSpeech layout:
#      wavs/<emotion>_<guid>.wav  22.05 kHz mono, RMS-normalised, ~150 ms of silence padded at both ends
#                                 (the original files are cut hard at the speech boundaries)
#      metadata.csv               <emotion>_<guid>|text  (the same 300 sentences in all 8 emotions)
#      dev_ids.txt                every emotion of 15 held-out sentences (120 ids), so no dev text is trained on
# 2. Remove room noise between words: the emotion embeddings would otherwise learn each session's hiss.
#    The quiet styles are the noisiest relative to their speech, so they get a stronger gate.
set -euo pipefail
D=data/thorsten_emotional
python -m piper_train.norm_audio.spectral_gate --input-dir $D/wavs --output-dir $D/wavs_clean \
  --max-db 15 --style-max-db "whisper=25,sleepy=25,drunk=25"
mkdir -p ${D}_clean && ln -sfn "$(realpath $D/wavs_clean)" ${D}_clean/wavs
cp $D/metadata.csv $D/dev_ids.txt ${D}_clean/

# 3. Train. (The published voice reused a fine-tune of thorsten/high on the uncleaned recordings and ran the
#    later stages on the cleaned ones; fine-tuning on the cleaned data directly, as here, is the simpler path.)
xvits-train \
  --dataset ${D}_clean \
  --base-ckpt "checkpoints/thorsten-high/epoch=2665-step=1182078.ckpt" \
  --base-config checkpoints/thorsten-high/config.json \
  --quality high \
  --dev-list ${D}_clean/dev_ids.txt \
  --style-regex '^([a-z]+)_' \
  --emotion-speakers \
  --style-gain "whisper=-16,sleepy=-4,drunk=-4" \
  --ft-epochs 300 --voice-epochs 400 \
  --out runs/thorsten_emotional \
  --compile --export
