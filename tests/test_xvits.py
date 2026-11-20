"""x-vits package: bundle arguments, export-friendly interpolation, MAS fallback."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


class BundleTest(unittest.TestCase):
    def test_roundtrip_and_synth_args(self):
        from xvits.bundle import Bundle

        with tempfile.TemporaryDirectory() as d:
            b = Bundle(name="v", config="c.json", voice="v.ckpt", predictor="p.pt", durations="d.pt",
                       prominence="s.pt", template="t.json", styles=["happy", "sad"])
            q = Bundle.load(b.save(Path(d) / "xvits.json"))
            self.assertEqual(q.styles, ["happy", "sad"])
            self.assertTrue(Path(q.voice).is_absolute())  # relative paths resolve next to xvits.json
            a = q.synth_args(style="happy", emphasis="predicted")
            self.assertIn("--prominence-student", a)
            self.assertEqual(a[a.index("--pp-temperature") + 1], "0.8,0.9,1.0")  # per-style temperature
            self.assertEqual(a[a.index("--noise-scale-w") + 1], "0.4")           # style SDP defaults
            self.assertNotIn("--pp-unvoiced-styles", a)
            self.assertNotIn("--prominence-student", q.synth_args(emphasis="marked"))
            self.assertNotIn("--emphasis-template", q.synth_args(emphasis="none"))
            w = Bundle(name="v", config="c", voice="v", predictor="p", styles=["whisper"],
                       defaults={"unvoiced_styles": ["whisper"]}).synth_args(style="label")
            self.assertEqual(w[w.index("--pp-unvoiced-styles") + 1], "whisper")

    def test_style_gain_and_speakers(self):
        from xvits.bundle import Bundle
        from xvits.recipe import parse_style_gain

        self.assertEqual(parse_style_gain("whisper=-16, sleepy=-4"), {"whisper": -16.0, "sleepy": -4.0})
        b = Bundle(name="v", config="c", voice="v", predictor="p", styles=["angry", "whisper"],
                   speakers=["angry", "whisper"], defaults={"style_gain_db": {"whisper": -16.0}})
        a = b.synth_args(style="label")
        self.assertEqual(a[a.index("--style-speakers") + 1], "angry,whisper")
        self.assertEqual(a[a.index("--style-gain") + 1], "whisper=-16.0")

    def test_spectral_gate(self):
        from piper_train.norm_audio.spectral_gate import spectral_gate

        sr, rng = 22050, np.random.default_rng(0)
        t = np.arange(sr) / sr
        tone = np.where((t > 0.3) & (t < 0.7), 0.3 * np.sin(2 * np.pi * 220 * t), 0.0)
        noise = 0.003 * rng.standard_normal(sr)
        pad = np.zeros(2000)
        x = np.concatenate([pad, tone + noise, pad]).astype(np.float32)
        y = spectral_gate(x, sr, max_db=20)
        q = slice(2000 + 1000, 2000 + 5000)  # a pause: noise only
        s = slice(2000 + 9000, 2000 + 14000)  # speech-like tone
        self.assertLess(np.std(y[q]), 0.5 * np.std(x[q]))                     # background reduced
        self.assertGreater(np.std(y[s]) / np.std(x[s]), 0.95)                   # tone kept
        self.assertLess(np.abs(y[:900]).max(), 1e-3)  # padding away from the STFT window's reach stays silent

    def test_per_channel(self):
        from xvits.runtime import per_channel

        self.assertEqual(per_channel("0.5"), [0.5] * 4)
        self.assertEqual(per_channel("0.8,0.7"), [0.8, 0.7, 0.7, 0.7])
        self.assertEqual(per_channel("0.8,0.7,1.0"), [0.8, 0.7, 0.7, 1.0])  # duration, pitch+voicing, energy


class InterpTest(unittest.TestCase):
    def test_vectorised_matches_reference(self):
        from xvits.export import interp_prosody_frames_onnx
        from piper_train.vits.models import interp_prosody_frames

        g = torch.Generator().manual_seed(0)
        for _ in range(5):
            t = int(torch.randint(2, 30, (1,), generator=g))
            pros = torch.randn(1, 3, t, generator=g)
            pros[:, 1] = (torch.rand(1, t, generator=g) > 0.4).float()
            w = torch.randint(0, 5, (1, 1, t), generator=g).float()
            nf = max(1, int(w.sum()))
            self.assertTrue(torch.allclose(interp_prosody_frames(pros, w, nf),
                                           interp_prosody_frames_onnx(pros, w, nf), atol=1e-5))


class MasTest(unittest.TestCase):
    def test_path_is_monotonic_and_complete(self):
        from piper_train.vits.monotonic_align import maximum_path

        neg = torch.randn(1, 12, 5)
        path = maximum_path(neg, torch.ones(1, 12, 5))[0].numpy()
        self.assertTrue((path.sum(1) == 1).all())              # one token per frame
        tok = path.argmax(1)
        self.assertTrue((np.diff(tok) >= 0).all() and tok[0] == 0 and tok[-1] == 4)


if __name__ == "__main__":
    unittest.main()
