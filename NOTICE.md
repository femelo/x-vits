# Third-party notices

| Component | Where | License |
|---|---|---|
| [Piper](https://github.com/rhasspy/piper) training code (Michael Hansen) | `src/piper_train/` (modified) | MIT |
| [VITS](https://github.com/jaywalnut310/vits) (Jaehyeon Kim) | `src/piper_train/vits/` (via Piper) | MIT |
| [Silero VAD](https://github.com/snakers4/silero-vad) model | `src/piper_train/norm_audio/models/silero_vad.onnx` | MIT |

Not included, but used at run time if you enable the evaluation stage:
[UTMOS](https://github.com/tarepan/SpeechMOS) (MIT; loaded from torch hub, or from a local
checkout given by `SPEECHMOS_DIR`) and
[Whisper](https://github.com/openai/whisper) via Hugging Face Transformers (MIT / Apache-2.0).

**Data and voices.** x-vits ships no audio and no trained voices. The results in the
README were obtained on the [Expresso](https://speechbot.github.io/expresso/)
corpus, which is licensed **CC BY-NC 4.0** (voices trained on it inherit the
non-commercial restriction), and on Thorsten Müller's
[emotional dataset](https://www.openslr.org/110/) with the
[Thorsten-Voice Piper-Scratch](https://huggingface.co/Thorsten-Voice/Piper-Scratch)
checkpoint, both **CC0**. Check the license of any dataset and base checkpoint you use.
