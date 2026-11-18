"""``xvits.json``: everything needed to synthesise with a trained x-vits voice."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Bundle:
    name: str
    config: str                      # Piper voice config.json (phoneme map, sample rate)
    voice: str                       # stage 2/5 checkpoint (prosody-conditioned VITS)
    predictor: str                   # stage 3 flow-matching predictor
    baseline: Optional[str] = None   # plain fine-tuned voice, for comparison
    durations: Optional[str] = None  # style-conditioned duration predictor
    prominence: Optional[str] = None # stage 4b student
    template: Optional[str] = None   # stage 4a stress-anchored emphasis template
    styles: List[str] = field(default_factory=list)
    frame_prosody: bool = False
    defaults: Dict = field(default_factory=dict)

    def __post_init__(self):
        d = {
            "noise_scale": 0.667,
            # tuned on Expresso: less timing randomness + slightly slower with the style predictor
            "noise_scale_w": 0.4 if self.durations else 0.8,
            "length_scale": 1.08 if self.durations else 1.0,
            "pp_temperature": "0.8,0.7",       # duration, pitch/energy
            "pp_guidance": "1.0,2.0",          # style exaggeration: timing, pitch/energy
            "style_temperature": {"happy": "0.8,0.9"},
            "template_scale": 1.5,
            "prominence_threshold": 0.5,
            "pp_steps": 16,
        }
        d.update(self.defaults or {})
        self.defaults = d

    # ---------------------------------------------------------------------
    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1))
        return path

    @classmethod
    def load(cls, path: Path) -> "Bundle":
        path = Path(path)
        d = json.loads(path.read_text())
        for k in ("config", "voice", "predictor", "baseline", "durations", "prominence", "template"):
            if d.get(k) and not Path(d[k]).is_absolute():
                d[k] = str((path.parent / d[k]).resolve())
        return cls(**d)

    # ---------------------------------------------------------------------
    def synth_args(self, style: Optional[str] = None, emphasis: str = "predicted") -> List[str]:
        """Flags for ``python -m piper_train.eval.synthesize``.

        style: a style name, 'label' (each eval row's 'category'), or None.
        emphasis: 'predicted' (student + template), 'marked' (template on the
        rows' *marks*), or 'none'.
        """
        d = self.defaults
        t = d["style_temperature"].get(style, d["pp_temperature"]) if style not in (None, "label") else d["pp_temperature"]
        a = ["--checkpoint", self.voice, "--prosody-predictor", self.predictor,
             "--pp-temperature", t, "--pp-guidance", d["pp_guidance"], "--pp-steps", str(d["pp_steps"]),
             "--pp-durations", "sdp", "--noise-scale", str(d["noise_scale"]),
             "--noise-scale-w", str(d["noise_scale_w"]), "--length-scale", str(d["length_scale"])]
        if style:
            a += ["--pp-style", style]
        if self.durations:
            a += ["--dp-style-ckpt", self.durations]
        if emphasis != "none" and self.template:
            a += ["--emphasis-template", self.template, "--emphasis-template-scale", str(d["template_scale"])]
            if emphasis == "predicted" and self.prominence:
                a += ["--prominence-student", self.prominence, "--prominence-threshold", str(d["prominence_threshold"])]
        return a
