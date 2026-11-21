"""``xvits-speak``: ONNX runtime for an exported x-vits voice (numpy + onnxruntime).

    xvits-speak --model voice_onnx/ --text "I *really* didn't expect that." \\
        --style happy --out hello.wav

Text may use *word* marks for explicit emphasis; without marks, the prominence
student decides (``--emphasis none`` turns emphasis off).
"""

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

N_TARGETS = 4


def per_channel(s: str, n_targets: int = N_TARGETS):
    """'a' | 'dur,rest' | 'dur,pitch,energy' | 'dur,pitch,energy,effort' ->
    [duration, lf0, vfrac, energy(, effort)]; effort follows energy unless given."""
    v = [float(x) for x in s.split(",")]
    if len(v) == 1:
        v = v * 3
    elif len(v) == 2:
        v = v + v[1:]
    dur, pitch, energy = v[:3]
    effort = v[3] if len(v) > 3 else energy
    return ([dur, pitch, pitch, energy] + [effort] * (n_targets - 4))[:n_targets]


class XVits:
    def __init__(self, model_dir: Path, providers=("CPUExecutionProvider",)):
        import onnxruntime as ort

        from piper_train.eval.phonemize import PiperPhonemizer

        self.dir = Path(model_dir)
        self.m = json.loads((self.dir / "runtime.json").read_text())
        f = self.m["files"]
        mk = lambda n: ort.InferenceSession(str(self.dir / f[n]), providers=list(providers))
        self.enc, self.dur, self.ac, self.pred = mk("encoder"), mk("durations"), mk("acoustic"), mk("predictor")
        self.prom = mk("prominence") if "prominence" in f else None
        self.template = json.loads((self.dir / f["template"]).read_text()) if "template" in f else None
        self.dp_cond = dict(np.load(self.dir / f["dp_cond"])) if "dp_cond" in f else None
        # multi-speaker voice: each style is also a speaker embedding (timbre)
        self.speakers = self.m.get("speakers")
        self.spk = dict(np.load(self.dir / f["speakers"])) if "speakers" in f else None
        self.pz = PiperPhonemizer(self.dir / self.m["config"])
        self.inv = {v: k for k, v in self.pz.id_map.items()}
        self.pstats = self.m["predictor"]["stats"]
        self.styles = self.m["predictor"]["styles"]
        self.n_targets = self.m["predictor"].get("n_targets", N_TARGETS)
        self.d = self.m["defaults"]

    @classmethod
    def from_pretrained(cls, repo_id: str, revision: Optional[str] = None, cache_dir: Optional[str] = None,
                        token: Optional[str] = None, providers=("CPUExecutionProvider",)) -> "XVits":
        """Load an exported voice from the Hugging Face Hub (or a local directory).

            voice = XVits.from_pretrained("fdemelo/x-vits")
        """
        if Path(repo_id).is_dir():
            return cls(Path(repo_id), providers)
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id, revision=revision, cache_dir=cache_dir, token=token,
            allow_patterns=["*.onnx", "*.json", "*.npz"],
        )
        return cls(Path(path), providers)

    @property
    def sample_rate(self) -> int:
        return int(self.m["sample_rate"])

    # ------------------------------------------------------------------
    def _style_index(self, style):
        if not style or not self.styles:
            return None
        if style not in self.styles:
            raise ValueError(f"unknown style {style!r}; this voice has {self.styles}")
        return 1 + self.styles.index(style)

    def _velocity(self, x_t, mask, cond, t, style):
        return self.pred.run(None, {"x_t": x_t, "mask": mask, "cond": cond,
                                    "t": np.array([t], np.float32), "style": np.array([style or 0], np.int64)})[0]

    def _sample_prosody(self, x, mask, style, temp, guidance, steps, rng):
        """OT-CFM ODE (cosine schedule, midpoint) with style guidance, as in
        ``ProsodyPredictor.sample``; returns (y [1, n_targets, T], initial noise)."""
        cond = x
        if self.m["predictor"]["token_feats"]:
            cond = np.concatenate([x, np.zeros((1, self.m["predictor"]["token_feats"], x.shape[2]), np.float32)], 1)
        noise0 = rng.standard_normal((1, self.n_targets, x.shape[2])).astype(np.float32)
        y = noise0 * temp * mask
        w = np.asarray(guidance, np.float32).reshape(1, -1, 1)
        guided = style is not None and bool((w != 1.0).any())

        def f(y_, t_):
            v = self._velocity(y_, mask, cond, t_, style)
            if guided:
                v = self._velocity(y_, mask, cond, t_, 0) + w * (v - self._velocity(y_, mask, cond, t_, 0))
            return v * mask

        ts = 1 - np.cos(0.5 * math.pi * np.linspace(0, 1, steps + 1, dtype=np.float32))
        for i in range(steps):
            t0, t1 = float(ts[i]), float(ts[i + 1])
            dt = t1 - t0
            k1 = f(y, t0)
            y = (y + dt * f(y + 0.5 * dt * k1, t0 + 0.5 * dt)) * mask
        return y, noise0

    def _decode(self, y, mask):
        vfrac = np.clip(y[:, 2:3], 0.0, 1.0)
        lf0 = y[:, 1:2] * (vfrac > 0.5)
        pros = np.concatenate([lf0, vfrac, y[:, 3:]], 1) * mask  # energy (, vocal effort)
        return self._bridge_effort(pros)

    def _bridge_effort(self, pros):
        """Effort comes from voiced tokens and is interpolated across the others (as in training)."""
        if pros.shape[1] < 4 or not self.pstats.get("tilt_voiced"):
            return pros
        from piper_train.prosody.features import voiced_effort

        pros[0, 3] = voiced_effort(pros[0, 3], pros[0, 1])
        return pros

    def _emphasis(self, text, x, mask, ids, style, scale, mode):
        """Per-token (pitch shift st, duration multiplier, log-energy offset, effort offset)
        from marks or the student."""
        from piper_train.prosody.emphasis import (apply_param_template, param_energy, parse_marked,
                                                  token_word_index, tonic_offsets)

        n = len(ids)
        none = (np.zeros(n, np.float32), np.ones(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32))
        if mode == "none" or self.template is None:
            return none
        plain, flags = parse_marked(text)
        t_ids, tok_w = token_word_index(plain, self.pz)
        if t_ids != ids:
            return none
        strength = np.ones(len(flags))
        if not any(flags) and mode == "auto" and self.prom is not None:
            logits = self.prom.run(None, {"x": x, "x_mask": mask})[0][0]
            nw = max(tok_w) + 1
            word = np.full(nw, -1e4, np.float32)
            for i, wi in enumerate(tok_w):
                if wi >= 0:
                    word[wi] = max(word[wi], logits[i])
            prob = 1 / (1 + np.exp(-word))
            thr = self.d["prominence_threshold"]
            strength = np.clip((prob - thr) / max(1e-6, 1 - thr), 0, 1)
            flags = list(strength > 0)
        offs = tonic_offsets(t_ids, tok_w, self.inv, flags)
        p_st, d_mul = apply_param_template(offs, self.template, scale, style,
                                           symbols=[self.inv.get(int(i), "?") for i in t_ids])
        tok_s = np.array([strength[w] if w >= 0 else 0.0 for w in tok_w], np.float32)
        e_log = param_energy(offs, self.template, scale, style) * tok_s
        t_log = param_energy(offs, self.template, scale, style, key="tilt") * tok_s
        return p_st * tok_s, np.power(d_mul, tok_s).astype(np.float32), e_log, t_log

    # ------------------------------------------------------------------
    def synthesize(self, text: str, style: Optional[str] = None, emphasis: str = "auto",
                   seed: Optional[int] = None, emphasis_scale: Optional[float] = None,
                   length_scale: Optional[float] = None, steps: Optional[int] = None,
                   style_strength: float = 1.0, guidance: Optional[str] = None,
                   temperature: Optional[str] = None, return_trace: bool = False):
        """Text (``*word*`` marks force emphasis) -> float32 audio at ``sample_rate``.

        style_strength: how far the style is exaggerated, relative to the voice's
            default: 1 = default, 0 = the style without exaggeration, 2 = twice the
            exaggeration.  Scales every channel of ``guidance`` around 1.
        guidance: per-channel style exaggeration "timing,pitch,energy,effort"
            (1 = none), overriding the voice's default; needs a style.
        temperature: per-channel variety "timing,pitch,energy[,effort]"; higher is
            livelier, lower is steadier.
        emphasis_scale: multiplier of the emphasis shape (pitch peak, lengthening,
            loudness) on emphasised words.
        steps: ODE steps of the prosody predictor (default 16; 4 is ~2.5x faster
            on CPU and changes pitch by ~0.1 semitone on average)."""
        d = self.d
        rng = np.random.default_rng(seed)
        plain = text.replace("*", "")
        ids = self.pz.ids(plain)
        x, mask = self.enc.run(None, {"ids": np.array([ids], np.int64), "lengths": np.array([len(ids)], np.int64)})
        sidx = self._style_index(style)
        temp_s = temperature or d["style_temperature"].get(style, d["pp_temperature"])
        temp = np.array(per_channel(temp_s, self.n_targets), np.float32).reshape(1, -1, 1)
        gg = [1.0 + style_strength * (g - 1.0) for g in per_channel(guidance or d["pp_guidance"], self.n_targets)]
        guidance = gg[0] if len(set(gg)) == 1 else gg
        n_steps = int(steps or d["pp_steps"])
        y, noise0 = self._sample_prosody(x, mask, sidx, temp, guidance, n_steps, rng)
        pros = self._decode(y, mask)
        voicing_fix = bool(d.get("voicing_fix")) and style not in d.get("unvoiced_styles", [])
        if voicing_fix:  # voicing follows the phonemes: no blips from flipped voicing
            from piper_train.prosody.features import fix_voicing_np

            pros[0] = fix_voicing_np(pros[0], [self.inv.get(int(i), "?") for i in ids])
            pros = self._bridge_effort(pros)
        p_st, d_mul, e_log, t_log = self._emphasis(text, x, mask, ids, style,
                                            d["template_scale"] if emphasis_scale is None else emphasis_scale,
                                            emphasis)
        shift = p_st * (math.log(2) / 12) / self.pstats["lf0_std"]
        pros[:, 0] = np.where(pros[:, 1] > 0.5, pros[:, 0] + shift[None], pros[:, 0])
        pros[:, 2] += e_log[None] / self.pstats["energy_std"]  # louder on emphasis, with the pitch peak
        if pros.shape[1] > 3:  # more vocal effort on emphasis
            pros[:, 3] += t_log[None] / self.pstats["tilt_std"]
        if style in d.get("unvoiced_styles", []):  # whispering: no voicing, no pitch
            pros[:, :2] = 0.0
        ceiling = d.get("pitch_ceiling", {}).get(style)
        if ceiling is not None:  # exaggeration must not leave the speaker's own pitch range
            from piper_train.prosody.features import soft_pitch_ceiling

            pros[0, 0] = soft_pitch_ceiling(pros[0, 0], pros[0, 1], ceiling)
        # durations
        ch = x.shape[1]  # the SDP width equals the text encoder width
        h_add = np.zeros((1, ch, len(ids)), np.float32)
        if self.dp_cond is not None:
            s = 0
            if style and style in self.m["durations_styles"]:
                s = 1 + self.m["durations_styles"].index(style)
            h_add += self.dp_cond["style"][s][None, :, None]
        spk = None
        if self.spk is not None:
            sp = self.speakers
            si = sp.index(style) if style in sp else (sp.index("neutral") if "neutral" in sp else 0)
            h_add += self.spk["dp_g"][si][None, :, None]
            spk = self.spk["g"][si][None, :, None].astype(np.float32)
        z = (rng.standard_normal((1, 2, len(ids))) * d["noise_scale_w"]).astype(np.float32)
        logw = self.dur.run(None, {"x": x, "x_mask": mask, "z": z, "h_add": h_add})[0]
        ls = d["length_scale"] if length_scale is None else length_scale
        w = np.exp(logw) * mask * d_mul.reshape(1, 1, -1)
        w_ceil = (np.ceil(w * ls) * mask).astype(np.float32)
        stress_cap = d.get("stress_max_frames")
        if stress_cap:  # a long stress mark sounds like an extra vowel
            from piper_train.prosody.features import cap_stress_frames

            w_ceil[0, 0] = cap_stress_frames(w_ceil[0, 0], [self.inv.get(int(i), "?") for i in ids],
                                             float(np.ceil(stress_cap * ls)))
        n_frames = int(w_ceil.sum())
        noise = (rng.standard_normal((1, self.m["inter_channels"], n_frames)) * d["noise_scale"]).astype(np.float32)
        feeds = {"x": x, "x_mask": mask, "prosody": pros.astype(np.float32), "w_ceil": w_ceil, "noise": noise}
        if spk is not None:
            feeds["spk"] = spk
        audio = self.ac.run(None, feeds)[0].squeeze()
        gain_db = d.get("style_gain_db", {}).get(style, 0.0)
        if gain_db:  # quieter whisper / sleepy / drunk: the training data was level-matched
            audio = audio * np.float32(10 ** (gain_db / 20))
        if return_trace:
            return audio, {"ids": ids, "noise0": noise0, "temperature": temp.ravel().tolist(), "guidance": guidance,
                           "steps": n_steps, "style_index": sidx, "prosody": pros, "p_st": p_st,
                           "e_log": e_log, "t_log": t_log, "voicing_fix": voicing_fix, "w_ceil": w_ceil,
                           "symbols": [self.inv.get(int(i), "?") for i in ids], "unvoiced": style in d.get("unvoiced_styles", []),
                           "d_mul": d_mul, "z": z, "h_add": h_add, "noise": noise, "length_scale": ls, "spk": spk, "gain_db": gain_db,
                           "pitch_ceiling": ceiling, "stress_cap": stress_cap}
        return audio


def main(argv=None):
    import soundfile as sf

    p = argparse.ArgumentParser(prog="xvits-speak", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="directory written by xvits-export, or a Hugging Face repo id (e.g. fdemelo/x-vits)")
    p.add_argument("--text", required=True, help="text; *word* marks force emphasis")
    p.add_argument("--style", help="style label (see runtime.json)")
    p.add_argument("--emphasis", choices=["auto", "marked", "none"], default="auto",
                   help="auto: marks if present, else predicted; marked: marks only")
    p.add_argument("--emphasis-scale", type=float)
    p.add_argument("--length-scale", type=float)
    p.add_argument("--steps", type=int, help="prosody ODE steps (default 16; 4 is faster on CPU)")
    p.add_argument("--style-strength", type=float, default=1.0,
                   help="style exaggeration relative to the voice default (0 none, 1 default, 2 double)")
    p.add_argument("--guidance", help='per-channel exaggeration "timing,pitch,energy,effort"')
    p.add_argument("--temperature", help='per-channel variety "timing,pitch,energy[,effort]"')
    p.add_argument("--seed", type=int)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    voice = XVits.from_pretrained(a.model)
    audio = voice.synthesize(a.text, a.style, a.emphasis, a.seed, a.emphasis_scale, a.length_scale, a.steps,
                             style_strength=a.style_strength, guidance=a.guidance, temperature=a.temperature)
    sf.write(a.out, audio, voice.sample_rate)
    print(f"wrote {a.out} ({len(audio) / voice.sample_rate:.2f} s)")


if __name__ == "__main__":
    main()
