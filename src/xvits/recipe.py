"""``xvits-train``: the full per-voice recipe, from an LJSpeech-style dataset and a
Piper checkpoint to an expressive voice bundle (and optionally ONNX).

Stages (each writes ``<stage>/done.json``; finished stages are skipped on rerun,
``--redo STAGE`` forces one):

  data        split into train/dev, strip *emphasis* marks, Piper preprocessing
  finetune    fine-tune the base checkpoint on the voice          (skippable)
  prosody     stage 1: per-token duration / pitch / energy / vocal-effort targets
  voice       stage 2(+5): fine-tune with prosody conditioning of the prior
  predictor   stage 3: flow-matching pitch/energy(/duration) predictor
  durations   style-conditioned duration predictor            (needs styles)
  prominence  stage 4b: word-prominence student from the recordings
  template    stage 4a: stress-anchored emphasis template
  bundle      write xvits.json (everything synthesis needs)
  eval        dev-set synthesis + metrics vs the fine-tuned baseline
  export      ONNX graphs + runtime manifest                    (optional)

Everything is single-speaker and single-language; styles and emphasis marks are
optional extras that switch on when the data has them.
"""

import argparse
import json
import logging
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

_LOG = logging.getLogger("xvits.recipe")
STAGES = ["data", "finetune", "prosody", "voice", "predictor", "durations",
          "prominence", "template", "bundle", "eval", "export"]


# ---------------------------------------------------------------------------
class Run:
    def __init__(self, args):
        self.a = args
        self.out: Path = args.out
        self.out.mkdir(parents=True, exist_ok=True)
        self.py = sys.executable

    def d(self, stage: str) -> Path:
        p = self.out / f"{STAGES.index(stage):02d}_{stage}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def done(self, stage: str) -> Optional[dict]:
        f = self.d(stage) / "done.json"
        if f.exists() and stage not in self.a.redo:
            return json.loads(f.read_text())
        return None

    def finish(self, stage: str, info: dict) -> dict:
        (self.d(stage) / "done.json").write_text(json.dumps(info, indent=1))
        return info

    def sh(self, stage: str, module: str, *args, env=None):
        cmd = [self.py, "-W", "ignore", "-m", module, *[str(x) for x in args]]
        log = self.d(stage) / "log.txt"
        _LOG.info("[%s] %s", stage, " ".join(cmd[3:]))
        t0 = time.time()
        for attempt in (1, 2):
            with open(log, "a", encoding="utf-8") as f:
                f.write("$ " + " ".join(cmd) + "\n")
                f.flush()
                r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
            # a crash on a signal (seen once inside libespeak-ng) is retried once
            if r.returncode >= 0 or attempt == 2:
                break
            _LOG.warning("[%s] %s crashed (signal %d), retrying once", stage, module, -r.returncode)
        if r.returncode:
            tail = "".join(log.read_text(encoding="utf-8", errors="replace").splitlines(True)[-25:])
            raise SystemExit(f"[{stage}] {module} failed (exit {r.returncode}); see {log}\n{tail}")
        _LOG.info("[%s] %s done in %.1f min", stage, module.split(".")[-1], (time.time() - t0) / 60)

    def train_args(self):
        a = self.a
        args = ["--batch-size", a.batch_size, "--accelerator", "gpu" if a.gpu else "cpu", "--devices", 1,
                "--precision", 32, "--quality", a.quality, "--log_every_n_steps", 20,
                "--num-workers", a.num_workers]
        if a.compile:
            args.append("--compile")
        return args


def last_ckpt(root: Path) -> Path:
    cks = sorted(root.glob("lightning_logs/version_*/checkpoints/*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not cks:
        raise SystemExit(f"no checkpoint under {root}")
    return cks[-1]


def data_language(config: str) -> str:
    """ISO language of the voice from its eSpeak voice ("en-us" -> "en", "de" -> "de")."""
    voice = json.loads(Path(config).read_text(encoding="utf-8")).get("espeak", {}).get("voice", "en-us")
    return voice.split("-")[0].split("_")[0]


def read_marks(path: Optional[Path]) -> Dict[str, str]:
    out = {}
    if path:
        for line in open(path, encoding="utf-8"):
            q = line.rstrip("\n").split("|")
            if len(q) >= 2:
                out[q[0]] = q[-1]
    return out


# ---------------------------------------------------------------------------
def stage_data(run: Run):
    if (info := run.done("data")):
        return info
    a, d = run.a, run.d("data")
    meta = [line.rstrip("\n").split("|") for line in open(a.dataset / "metadata.csv", encoding="utf-8") if line.strip()]
    marks = read_marks(a.marked)
    for row in meta:  # marks may also live directly in metadata.csv
        if "*" in row[-1]:
            marks.setdefault(row[0], row[-1])
    wav_dir = a.dataset / ("wavs" if (a.dataset / "wavs").is_dir() else "wav")
    ids = [r[0] for r in meta if (wav_dir / f"{r[0]}.wav").exists()]
    text = {r[0]: r[-1].replace("*", "") for r in meta}
    if a.dev_list:
        dev = {l.strip().split("|")[0] for l in open(a.dev_list, encoding="utf-8") if l.strip()}
    else:
        rnd = random.Random(a.seed)
        n_dev = a.dev_size or max(20, min(150, len(ids) // 20))
        dev = set(rnd.sample(ids, n_dev))
    for split, keep in (("train", [i for i in ids if i not in dev]), ("dev", [i for i in ids if i in dev])):
        sd = d / f"{split}_raw"
        sd.mkdir(exist_ok=True)
        link = sd / "wavs"
        if not link.exists():
            link.symlink_to(wav_dir.resolve())
        (sd / "metadata.csv").write_text("".join(f"{i}|{text[i]}\n" for i in keep), encoding="utf-8")
        run.sh("data", "piper_train.preprocess_local", "--input-dir", sd, "--output-dir", d / split,
               "--config", a.base_config, "--max-workers", a.cpu_workers)
    if marks:
        (d / "marks.csv").write_text("".join(f"{k}|{v}\n" for k, v in marks.items()), encoding="utf-8")
    # dev eval set (id, utt, category, text, phoneme_ids [, marked])
    rx = re.compile(a.style_regex) if a.style_regex else None
    rows = []
    for r in map(json.loads, open(d / "dev" / "dataset.jsonl", encoding="utf-8")):
        utt = Path(r["audio_path"]).stem
        cat = (rx.search(utt).group(1) if rx and rx.search(utt) else "default")
        row = {"id": f"{len(rows):04d}", "utt": utt, "category": cat, "text": r["text"], "phoneme_ids": r["phoneme_ids"]}
        if utt in marks and "*" in marks[utt]:
            row["marked"] = marks[utt]
        rows.append(row)
    (d / "dev_set.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    shutil.copy(a.base_config, d / "config.json")
    n_train = sum(1 for _ in open(d / "train" / "dataset.jsonl"))
    return run.finish("data", {"train": str(d / "train"), "dev": str(d / "dev"), "dev_set": str(d / "dev_set.jsonl"),
                               "config": str(d / "config.json"), "marks": str(d / "marks.csv") if marks else None,
                               "n_train": n_train, "n_dev": len(rows), "n_marked": len(marks)})


def stage_finetune(run: Run, data):
    if (info := run.done("finetune")):
        return info
    a, d = run.a, run.d("finetune")
    if a.skip_finetune:
        return run.finish("finetune", {"ckpt": str(a.base_ckpt), "skipped": True})
    run.sh("finetune", "piper_train", "--dataset-dir", data["train"], "--pretrained-checkpoint", a.base_ckpt,
           "--max_epochs", a.ft_epochs, "--checkpoint-epochs", a.ft_epochs, "--default_root_dir", d, *run.train_args())
    return run.finish("finetune", {"ckpt": str(last_ckpt(d))})


def stage_prosody(run: Run, data, ft):
    if (info := run.done("prosody")):
        return info
    a, d = run.a, run.d("prosody")
    for split in ("train", "dev"):
        run.sh("prosody", "piper_train.prosody.extract", "--dataset-dir", data[split], "--checkpoint", ft["ckpt"],
               "--max-workers", a.cpu_workers, "--out", d / f"{split}.pt")
    import torch  # merge: dev targets, train normalisation stats

    tr = torch.load(d / "train.pt", weights_only=False)
    dv = torch.load(d / "dev.pt", weights_only=False)
    assert not set(tr["utts"]) & set(dv["utts"])
    tr["utts"].update(dv["utts"])
    torch.save(tr, d / "prosody_all.pt")
    run.sh("prosody", "piper_train.prosody.make_dataset", "--dataset-dir", data["train"],
           "--prosody", d / "train.pt", "--out-dir", d / "train_prosody")
    return run.finish("prosody", {"all": str(d / "prosody_all.pt"), "train_dataset": str(d / "train_prosody"),
                                  "f0_range": tr["stats"].get("f0_range")})


def stage_voice(run: Run, pros, ft):
    if (info := run.done("voice")):
        return info
    a, d = run.a, run.d("voice")
    extra = ["--prosody-frame"] if a.frame_prosody else []
    cfg = json.loads((Path(pros["train_dataset"]) / "config.json").read_text(encoding="utf-8"))
    dataset, init, speakers = pros["train_dataset"], ["--pretrained-checkpoint", ft["ckpt"]], None
    if a.emotion_speakers:
        # each style also gets its own speaker embedding (timbre: whisper, shouting), learned from a
        # zero start on top of the single-speaker voice
        dataset, speakers = emotion_speaker_dataset(Path(pros["train_dataset"]), d / "dataset_speakers", a.style_regex)
        init = ["--resume_from_single_speaker_checkpoint", ft["ckpt"], "--zero-init-speaker-cond"]
    run.sh("voice", "piper_train", "--dataset-dir", dataset, *init,
           "--prosody-channels", cfg["prosody"]["channels"], *extra, "--max_epochs", a.voice_epochs, "--checkpoint-epochs", a.voice_epochs,
           "--default_root_dir", d, *run.train_args())
    return run.finish("voice", {"ckpt": str(last_ckpt(d)), "frame_prosody": bool(a.frame_prosody), "speakers": speakers})


def emotion_speaker_dataset(src: Path, dst: Path, style_regex: str):
    """Copy of a prosody dataset dir where each style is a speaker (sorted style names)."""
    rx = re.compile(style_regex)
    rows = [json.loads(l) for l in open(src / "dataset.jsonl", encoding="utf-8") if l.strip()]
    style = lambda r: rx.search(Path(r["audio_path"]).stem).group(1)
    speakers = sorted({style(r) for r in rows})
    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / "dataset.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            r["speaker_id"] = speakers.index(style(r))
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    cfg["num_speakers"] = len(speakers)
    cfg["speaker_id_map"] = {s: i for i, s in enumerate(speakers)}
    (dst / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(dst), speakers


def build_emphasis_db(run: Run, data) -> Optional[Path]:
    """Stress-anchored emphasis features for marked utterances (zeros elsewhere)."""
    if not data.get("marks"):
        return None
    out = run.d("predictor") / "emphasis_tonic.pt"
    if out.exists():
        return out
    import numpy as np
    import torch

    from piper_train.eval.phonemize import PiperPhonemizer
    from piper_train.prosody.emphasis import tonic_features

    pz = PiperPhonemizer(Path(data["config"]))
    marks = read_marks(Path(data["marks"]))
    db = {}
    for split in ("train", "dev"):
        for r in map(json.loads, open(Path(data[split]) / "dataset.jsonl", encoding="utf-8")):
            u = Path(r["audio_path"]).stem
            e = np.zeros((3, len(r["phoneme_ids"])), np.float32)
            if "*" in marks.get(u, ""):
                ids, f = tonic_features(marks[u], pz)
                if ids == r["phoneme_ids"]:
                    e = f
            db[u] = e
    torch.save({"emphasis": db, "channels": ["flag", "tonic_offset/12", "near_tonic"]}, out)
    return out


def stage_predictor(run: Run, data, pros, voice):
    if (info := run.done("predictor")):
        return info
    a, d = run.a, run.d("predictor")
    extra = []
    if a.style_regex:
        extra += ["--style-regex", a.style_regex]
    emph = build_emphasis_db(run, data)
    if emph:
        extra += ["--emphasis", emph]
    run.sh("predictor", "piper_train.prosody.train_predictor", "--checkpoint", voice["ckpt"], "--prosody", pros["all"],
           "--dataset", Path(pros["train_dataset"]) / "dataset.jsonl", "--eval-set", data["dev_set"],
           "--out-dir", d, "--steps", a.predictor_steps, *extra)
    return run.finish("predictor", {"ckpt": str(d / "best.pt"), "emphasis_db": str(emph) if emph else None})


def stage_durations(run: Run, data, pros, voice, pred):
    if (info := run.done("durations")):
        return info
    a, d = run.a, run.d("durations")
    if not a.style_regex:
        return run.finish("durations", {"ckpt": None, "reason": "no styles: the voice's own duration predictor is used"})
    extra = ["--emphasis", pred["emphasis_db"]] if pred.get("emphasis_db") else []
    if voice.get("speakers"):
        extra += ["--style-speakers", ",".join(voice["speakers"])]
    run.sh("durations", "piper_train.prosody.train_dp_style", "--checkpoint", voice["ckpt"], "--prosody", pros["all"],
           "--dataset", Path(pros["train_dataset"]) / "dataset.jsonl", "--eval-set", data["dev_set"],
           "--style-regex", a.style_regex, "--steps", a.durations_steps, "--out-dir", d, *extra)
    ck = d / "best.pt"
    return run.finish("durations", {"ckpt": str(ck) if ck.exists() else None})


def stage_prominence(run: Run, data, pros, voice):
    if (info := run.done("prominence")):
        return info
    a, d = run.a, run.d("prominence")
    extra = ["--marked", data["marks"]] if data.get("marks") else []
    run.sh("prominence", "piper_train.prosody.train_prominence", "--checkpoint", voice["ckpt"], "--prosody", pros["all"],
           "--dataset", Path(pros["train_dataset"]) / "dataset.jsonl", "--eval-set", data["dev_set"],
           "--config", data["config"], "--out-dir", d, "--steps", a.prominence_steps, *extra)
    return run.finish("prominence", {"ckpt": str(d / "best.pt")})


def stage_template(run: Run, data, pros):
    if (info := run.done("template")):
        return info
    a, d = run.a, run.d("template")
    extra = ["--marked", data["marks"]] if data.get("marks") else []
    if a.style_regex:
        extra += ["--style-regex", a.style_regex]
    run.sh("template", "piper_train.prosody.fit_template", "--prosody", pros["all"],
           "--dataset", Path(pros["train_dataset"]) / "dataset.jsonl", "--config", data["config"],
           "--out", d / "emphasis_template.json", *extra)
    return run.finish("template", {"json": str(d / "emphasis_template.json")})


def parse_style_gain(s: str) -> Dict[str, float]:
    """'whisper=-8,sleepy=-4' -> {'whisper': -8.0, 'sleepy': -4.0}"""
    return {k.strip(): float(v) for k, v in (kv.split("=") for kv in s.split(",") if kv.strip())}


def voice_limits(prosody_pt: str, dataset_jsonl: Path, config: str, style_regex: Optional[str]) -> dict:
    """Safeguards measured on the voice's own training alignments:
    - stress_max_frames: p95 duration of stress-mark tokens (a long one is heard as an extra vowel);
    - pitch_ceiling: per style, p95 of standardised log-F0, soft-limited above (style exaggeration
      otherwise holds a vowel above anything the speaker produced; p95 sounded better than p99)."""
    import numpy as np
    import torch

    P = torch.load(prosody_pt, map_location="cpu", weights_only=False)
    utts, st = P["utts"], P["stats"]
    id_map = json.loads(Path(config).read_text(encoding="utf-8"))["phoneme_id_map"]
    stress_ids = {v[0] for k, v in id_map.items() if k in ("ˈ", "ˌ")}
    stress = []
    for line in open(dataset_jsonl, encoding="utf-8"):
        r = json.loads(line)
        u = utts.get(Path(r["audio_path"]).stem)
        if u is not None and len(u["dur"]) == len(r["phoneme_ids"]):
            stress += [int(d) for t, d in zip(r["phoneme_ids"], u["dur"]) if t in stress_ids]
    out = {"stress_max_frames": float(np.percentile(stress, 95)) if stress else None}
    if style_regex:
        rx, by = re.compile(style_regex), {}
        for k, u in utts.items():
            m = rx.search(k)
            l = np.asarray(u["lf0"], float)
            l = l[~np.isnan(l)]
            if m and len(l):
                by.setdefault(m.group(1), []).append((l - st["lf0_mean"]) / st["lf0_std"])
        out["pitch_ceiling"] = {s: round(float(np.percentile(np.concatenate(v), 95)), 3) for s, v in sorted(by.items())}
    _LOG.info("voice limits: %s", out)
    return out


def unvoiced_styles(prosody_pt: str, style_regex: str, ratio: float = 0.2) -> List[str]:
    """Styles whose recordings are (nearly) unvoiced, e.g. whispering: duration-weighted
    voiced fraction below ``ratio`` x the median over styles. Rendering them with
    voicing forced off avoids a half-voiced buzz."""
    import numpy as np
    import torch

    rx, acc = re.compile(style_regex), {}
    for utt, v in torch.load(prosody_pt, map_location="cpu", weights_only=False)["utts"].items():
        m = rx.search(utt)
        if m:
            d = np.asarray(v["dur"], float)
            s = acc.setdefault(m.group(1), [0.0, 0.0])
            s[0] += float((np.asarray(v["vfrac"], float) * d).sum())
            s[1] += float(d.sum())
    frac = {k: a / max(b, 1e-9) for k, (a, b) in acc.items()}
    if len(frac) < 2:
        return []
    med = float(np.median(list(frac.values())))
    out = sorted(k for k, f in frac.items() if f < ratio * med)
    _LOG.info("voiced fraction per style: %s -> unvoiced: %s",
              {k: round(f, 3) for k, f in sorted(frac.items())}, out or "none")
    return out


def stage_bundle(run: Run, data, ft, pros, voice, pred, durs, prom, tpl):
    if (info := run.done("bundle")):
        return info
    from .bundle import Bundle

    a, d = run.a, run.d("bundle")
    styles, defaults = [], {}
    if a.style_regex:
        import torch

        styles = list(torch.load(pred["ckpt"], map_location="cpu", weights_only=False).get("styles") or [])
        defaults["unvoiced_styles"] = unvoiced_styles(str(Path(pros["all"]).parent / "train.pt"), a.style_regex)
        if a.style_gain:
            defaults["style_gain_db"] = parse_style_gain(a.style_gain)
    defaults.update(voice_limits(str(Path(pros["all"]).parent / "train.pt"),
                                 Path(pros["train_dataset"]) / "dataset.jsonl", data["config"], a.style_regex))
    b = Bundle(
        name=a.name or a.out.name, config=data["config"], voice=voice["ckpt"], baseline=ft["ckpt"],
        predictor=pred["ckpt"], durations=durs.get("ckpt"), prominence=prom["ckpt"], template=tpl["json"],
        styles=styles, speakers=voice.get("speakers"), frame_prosody=voice.get("frame_prosody", False), defaults=defaults,
    )
    path = b.save(d / "xvits.json")
    return run.finish("bundle", {"bundle": str(path)})


def stage_eval(run: Run, data, pros, bundle):
    if (info := run.done("eval")):
        return info
    from .bundle import Bundle

    a, d = run.a, run.d("eval")
    b = Bundle.load(Path(bundle["bundle"]))
    systems = {"baseline": ["--checkpoint", b.baseline],
               "x-vits": b.synth_args(style="label", emphasis="predicted")}
    report = {}
    for name, args in systems.items():
        o = d / name
        run.sh("eval", "piper_train.eval.synthesize", "--eval-set", data["dev_set"], "--config", data["config"],
               "--out-dir", o, "--device", "cuda" if a.gpu else "cpu", *args)
        run.sh("eval", "piper_train.eval.score_prosody", "--wav-dir", o, "--eval-set", data["dev_set"],
               "--prosody", pros["all"], "--checkpoint", b.baseline)
        m = json.loads((o / "prosody.json").read_text())
        rep = {k: m[k] for k in ("f0_corr", "f0_rmse_st", "f0_std_ratio", "dur_corr", "rate_ratio") if k in m}
        if not a.no_mos:
            run.sh("eval", "piper_train.eval.score_utmos", "--wav-dir", o, "--device", "cuda" if a.gpu else "cpu")
            run.sh("eval", "piper_train.eval.score_wer", "--wav-dir", o, "--device", "cuda" if a.gpu else "cpu",
                   "--language", data_language(data["config"]))
            rep["utmos"] = json.loads((o / "utmos.json").read_text())["utmos_mean"]
            rep["wer"] = json.loads((o / "wer.json").read_text())["wer"]
        report[name] = rep
    lines = ["| system | " + " | ".join(report["baseline"]) + " |", "|---" * (1 + len(report["baseline"])) + "|"]
    for n, r in report.items():
        lines.append(f"| {n} | " + " | ".join(f"{v:.3f}" for v in r.values()) + " |")
    (d / "report.md").write_text(f"# {b.name}: dev set ({data['n_dev']} utterances)\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))
    return run.finish("eval", {"report": str(d / "report.md"), **report})


def stage_export(run: Run, bundle):
    if (info := run.done("export")):
        return info
    d = run.d("export")
    run.sh("export", "xvits.export", bundle["bundle"], "--out", d / "onnx", "--verify")
    return run.finish("export", {"onnx": str(d / "onnx")})


# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(prog="xvits-train", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True, help="LJSpeech-style dir: metadata.csv (id|...|text) + wavs/")
    p.add_argument("--base-ckpt", type=Path, required=True, help="Piper/VITS .ckpt to start from")
    p.add_argument("--base-config", type=Path, help="its config.json (phoneme map); default: next to the checkpoint")
    p.add_argument("--out", type=Path, required=True, help="run directory")
    p.add_argument("--name", help="voice name (default: run directory name)")
    p.add_argument("--style-regex", help="regex whose group 1 extracts a style label from the utterance id")
    p.add_argument("--marked", type=Path, help="id|...|text with *emphasis* marks (optional)")
    p.add_argument("--dev-list", type=Path, help="ids to hold out (default: random ~5%%, 20-150)")
    p.add_argument("--dev-size", type=int)
    p.add_argument("--skip-finetune", action="store_true", help="--base-ckpt is already fine-tuned on this voice")
    p.add_argument("--ft-epochs", type=int, default=400)
    p.add_argument("--voice-epochs", type=int, default=200)
    p.add_argument("--style-gain", help='output level per style in dB, e.g. "whisper=-8,sleepy=-4,drunk=-4"')
    p.add_argument("--emotion-speakers", action="store_true",
                   help="also give each style its own speaker embedding in the voice (timbre: whisper, shouting)")
    p.add_argument("--no-frame-prosody", dest="frame_prosody", action="store_false",
                   help="stage 2 only (per-token prosody, no frame-level interpolation)")
    p.add_argument("--predictor-steps", type=int, default=10000)
    p.add_argument("--durations-steps", type=int, default=3000)
    p.add_argument("--prominence-steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--quality", default="medium", choices=["x-low", "medium", "high"])
    p.add_argument("--cpu", dest="gpu", action="store_false")
    p.add_argument("--compile", action="store_true", help="torch.compile the VITS fine-tunes")
    p.add_argument("--num-workers", type=int, default=8, help="dataloader workers")
    p.add_argument("--cpu-workers", type=int, default=4, help="processes for preprocessing/extraction")
    p.add_argument("--no-mos", action="store_true", help="skip UTMOS/Whisper in eval (no model downloads)")
    p.add_argument("--export", action="store_true", help="also export ONNX at the end")
    p.add_argument("--until", choices=STAGES, help="stop after this stage")
    p.add_argument("--redo", action="append", default=[], choices=STAGES, help="rerun a finished stage")
    p.add_argument("--quick", action="store_true", help="smoke test: tiny epochs/steps")
    p.add_argument("--seed", type=int, default=1234)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    # stage records and the bundle store paths; keep them valid from any working directory
    for k in ("dataset", "base_ckpt", "base_config", "out", "marked", "dev_list"):
        if getattr(a, k) is not None:
            setattr(a, k, getattr(a, k).resolve())
    if a.base_config is None:
        a.base_config = a.base_ckpt.parent / "config.json"
        if not a.base_config.exists():
            p.error("--base-config is required (no config.json next to the checkpoint)")
    if a.quick:
        a.ft_epochs, a.voice_epochs = min(a.ft_epochs, 2), min(a.voice_epochs, 2)
        a.predictor_steps, a.durations_steps, a.prominence_steps = 400, 200, 250
        a.no_mos = True

    run = Run(a)
    stop = STAGES.index(a.until) if a.until else len(STAGES) - 1

    def go(stage):
        return STAGES.index(stage) <= stop

    data = stage_data(run)
    if not go("finetune"): return
    ft = stage_finetune(run, data)
    if not go("prosody"): return
    pros = stage_prosody(run, data, ft)
    if not go("voice"): return
    voice = stage_voice(run, pros, ft)
    if not go("predictor"): return
    pred = stage_predictor(run, data, pros, voice)
    if not go("durations"): return
    durs = stage_durations(run, data, pros, voice, pred)
    if not go("prominence"): return
    prom = stage_prominence(run, data, pros, voice)
    if not go("template"): return
    tpl = stage_template(run, data, pros)
    if not go("bundle"): return
    bundle = stage_bundle(run, data, ft, pros, voice, pred, durs, prom, tpl)
    if not go("eval"): return
    stage_eval(run, data, pros, bundle)
    if a.export and go("export"):
        stage_export(run, bundle)
    _LOG.info("bundle: %s", bundle["bundle"])


if __name__ == "__main__":
    main()
