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
            self.assertEqual(a[a.index("--pp-temperature") + 1], "0.8,0.9")  # per-style temperature
            self.assertEqual(a[a.index("--noise-scale-w") + 1], "0.4")       # style SDP defaults
            self.assertNotIn("--prominence-student", q.synth_args(emphasis="marked"))
            self.assertNotIn("--emphasis-template", q.synth_args(emphasis="none"))


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
