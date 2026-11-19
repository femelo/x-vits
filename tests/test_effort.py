"""Vocal-effort (spectral tilt) channel: extraction, features, predictor width."""

import unittest

import numpy as np
import torch

from piper_train.prosody.features import feature_matrix, n_channels
from piper_train.prosody.predictor import ProsodyPredictor

STATS3 = {"lf0_mean": 5.3, "lf0_std": 0.22, "energy_mean": 2.6, "energy_std": 1.2,
          "logdur_mean": 1.17, "logdur_std": 0.47}
STATS4 = dict(STATS3, tilt_mean=-4.0, tilt_std=1.5)


def _utt(n=6):
    rng = np.random.default_rng(1)
    return {"dur": rng.integers(1, 9, n).astype(np.int16), "lf0": rng.normal(5.3, 0.2, n).astype(np.float32),
            "vfrac": np.ones(n, np.float32), "energy": rng.normal(2.6, 1, n).astype(np.float32),
            "tilt": rng.normal(-4, 1.5, n).astype(np.float32)}


class EffortTest(unittest.TestCase):
    def test_tilt_tracks_brightness(self):
        from piper_train.prosody.extract import token_features

        sr, n_bins, frames = 22050, 513, 20
        f = np.linspace(0, sr / 2, n_bins)
        dull = np.exp(-f / 500.0)                    # energy mostly below 1 kHz
        bright = dull + 0.3 * ((f > 2000) & (f < 5000))
        spec = torch.from_numpy(np.stack([dull] * 10 + [bright] * 10, 1)).float()
        dur = np.array([10, 10])
        _, _, _, tilt = token_features(dur, np.zeros(frames), spec, sr)
        self.assertGreater(tilt[1], tilt[0] + 2.0)

    def test_feature_channels(self):
        u = _utt()
        self.assertEqual(n_channels(STATS3), 3)
        self.assertEqual(feature_matrix(u, STATS3).shape, (6, 3))  # older extractions unchanged
        self.assertEqual(n_channels(STATS4), 4)
        m = feature_matrix(u, STATS4)
        self.assertEqual(m.shape, (6, 4))
        np.testing.assert_allclose(m[:, 3], (u["tilt"] + 4.0) / 1.5, rtol=1e-5)

    def test_predictor_width(self):
        kw = dict(text_channels=16, hidden=32, n_wn_layers=2, n_attn_layers=1, n_heads=2)
        self.assertEqual(ProsodyPredictor(STATS3, **kw).n_targets, 4)
        p = ProsodyPredictor(STATS4, **kw).eval()
        self.assertEqual(p.n_targets, 5)
        x, m = torch.randn(1, 16, 6), torch.ones(1, 1, 6)
        y = p.sample(x, m, n_steps=2)
        self.assertEqual(tuple(y.shape), (1, 5, 6))
        dur, pros = p.decode(y, m)
        self.assertEqual(tuple(pros.shape), (1, 4, 6))  # lf0, vfrac, energy, effort
        t = torch.from_numpy(p.encode_targets(_utt())).t().unsqueeze(0)
        self.assertEqual(tuple(t.shape), (1, 5, 6))
        p.train().loss(x, t, m).backward()


if __name__ == "__main__":
    unittest.main()
