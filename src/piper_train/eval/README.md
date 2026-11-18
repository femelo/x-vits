# Eval / test harness (en_US-lessac-medium)

CPU by default; nothing here touches the GPU unless you pass `--device cuda`.

```sh
cd src/python
python -m piper_train.eval.fetch_voice --out-dir ../../../data/ljspeech-medium [--ckpt]   # --ckpt = 845 MB
python -m unittest discover -s tests -t .                                          # unit + CPU integration tests

# baseline vs CFM (same eval set, same seed)
python -m piper_train.eval.synthesize --checkpoint lessac.ckpt --out-dir out/baseline
python -m piper_train.eval.score_utmos --wav-dir out/baseline
python -m piper_train.eval.score_wer   --wav-dir out/baseline
```

The eval sentences come from lessac's own training manifest (text + phoneme ids), so
numbers are good for A/B comparison, not for generalisation claims.

Fine-tune from the VITS checkpoint (when you're ready for the GPU):

```sh
python -m piper_train --dataset-dir <dir> --batch-size 16 --accelerator gpu --devices 1 \
  --pretrained-checkpoint lessac.ckpt --use-cfm --flow-mas-warmup-steps 2000
```
