"""Stage 3 flow-matching prosody predictor."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from piper_train.prosody.predictor import N_TARGETS, ProsodyPredictor

STATS = {
    "lf0_mean": 5.3, "lf0_std": 0.22, "energy_mean": 2.6, "energy_std": 1.2,
    "logdur_mean": 1.17, "logdur_std": 0.47,
}


def _pred(**kw):
    return ProsodyPredictor(STATS, text_channels=16, hidden=32, n_wn_layers=2,
                            n_attn_layers=1, n_heads=2, **kw).eval()


def _utt(n=7):
    rng = np.random.default_rng(0)
    lf0 = rng.normal(5.3, 0.2, n).astype(np.float32)
    lf0[[1, 4]] = np.nan  # unvoiced
    vfrac = np.ones(n, np.float32)
    vfrac[[1, 4]] = 0.0
    return {"dur": rng.integers(1, 20, n).astype(np.int16), "lf0": lf0,
            "vfrac": vfrac, "energy": rng.normal(2.6, 1.0, n).astype(np.float32)}


class PredictorTest(unittest.TestCase):
    def test_encode_decode_roundtrip(self):
        from piper_train.prosody.features import feature_matrix

        p, u = _pred(), _utt()
        y = torch.from_numpy(p.encode_targets(u)).t().unsqueeze(0)
        self.assertEqual(tuple(y.shape), (1, N_TARGETS, 7))
        dur, pros = p.decode(y, torch.ones(1, 1, 7))
        self.assertTrue(torch.equal(dur[0], torch.from_numpy(u["dur"].astype(np.float32))))
        ref = torch.from_numpy(feature_matrix(u, STATS)).t()
        self.assertTrue(torch.allclose(pros[0], ref, atol=1e-5))

    def test_untrained_velocity_zero_and_masked(self):
        p = _pred()
        x = torch.randn(2, 16, 9)
        mask = torch.ones(2, 1, 9)
        mask[1, :, 5:] = 0
        v = p.cfm.estimator(torch.randn(2, N_TARGETS, 9), mask, x, torch.rand(2))
        self.assertEqual(torch.count_nonzero(v).item(), 0)
        y = p.sample(x, mask, n_steps=2)
        self.assertEqual(torch.count_nonzero(y[1, :, 5:]).item(), 0)
        dur, _ = p.decode(y, mask)
        self.assertTrue((dur[0] >= 1).all() and (dur[1, 5:] == 0).all())

    def test_loss_backprop_and_save_load(self):
        p = _pred().train()
        x, y = torch.randn(2, 16, 9), torch.randn(2, N_TARGETS, 9)
        p.loss(x, y, torch.ones(2, 1, 9)).backward()
        self.assertGreater(p.cfm.estimator.out.weight.grad.abs().sum().item(), 0)
        with tempfile.TemporaryDirectory() as d:
            p.save(Path(d) / "p.pt")
            q = ProsodyPredictor.load(Path(d) / "p.pt")
        self.assertEqual((q.kw, q.stats), (p.kw, p.stats))
        for a, b in zip(p.state_dict().values(), q.state_dict().values()):
            self.assertTrue(torch.equal(a, b))

    def test_style_conditioning(self):
        p = _pred(styles=["happy", "sad"]).train()
        self.assertEqual([p.style_index(n) for n in (None, "happy", "sad")], [0, 1, 2])
        with self.assertRaises(ValueError):
            p.style_index("angry")
        torch.nn.init.normal_(p.cfm.estimator.out.weight)  # zero-init blocks gradients/velocity
        x, y, m = torch.randn(2, 16, 9), torch.randn(2, N_TARGETS, 9), torch.ones(2, 1, 9)
        p.loss(x, y, m, style=torch.LongTensor([1, 0])).backward()
        self.assertGreater(p.cfm.estimator.style.weight.grad.abs().sum().item(), 0)
        p.eval()
        g = torch.LongTensor([1, 2])
        torch.manual_seed(0); a = p.sample(x, m, n_steps=2, style=g)
        torch.manual_seed(0); b = p.sample(x, m, n_steps=2, style=torch.LongTensor([2, 1]))
        torch.manual_seed(0); c = p.sample(x, m, n_steps=2, style=g, guidance=2.0)
        self.assertFalse(torch.allclose(a, b) or torch.allclose(a, c))
        with tempfile.TemporaryDirectory() as d:
            p.save(Path(d) / "p.pt")
            self.assertEqual(ProsodyPredictor.load(Path(d) / "p.pt").styles, ["happy", "sad"])

    def test_per_channel_guidance_and_shape_durations(self):
        from piper_train.prosody.predictor import shape_durations

        p = _pred(styles=["happy"])
        torch.nn.init.normal_(p.cfm.estimator.out.weight)
        x, m, g = torch.randn(1, 16, 9), torch.ones(1, 1, 9), torch.LongTensor([1])
        torch.manual_seed(0); a = p.sample(x, m, n_steps=2, style=g)
        torch.manual_seed(0); b = p.sample(x, m, n_steps=2, style=g, guidance=[1.0, 2.0, 2.0, 2.0])
        self.assertFalse(torch.allclose(a[:, 1:], b[:, 1:]))
        # tokens: ^ _ a _ b _ " " _ c _ . _ $   (a, b, c phonemes)
        sym = ["^", "_", "a", "_", "b", "_", " ", "_", "c", "_", ".", "_", "$"]
        m2s = dict(enumerate(sym))
        dur = torch.full((1, len(sym)), 30.0)
        out = shape_durations(dur, [list(range(len(sym)))], m2s, max_frames=10, final_stretch=1.5)
        # capped: phonemes and in-word pads; kept: pads next to space/punct, punct, BOS/EOS
        self.assertEqual(out[0, [2, 3, 4]].tolist(), [10.0, 10.0, 10.0])
        self.assertEqual(out[0, [5, 6]].tolist(), [30.0, 30.0])
        # last word (tokens after the space): pause pads 30 -> 45, c capped to 10 then 15
        self.assertEqual(out[0, 7].item(), 45.0)
        self.assertEqual(out[0, 8].item(), 15.0)
        self.assertEqual(out[0, 9].item(), 45.0)

    def test_position_features(self):
        p = _pred(pos_features=True)
        torch.nn.init.normal_(p.cfm.estimator.out.weight)
        mask = torch.ones(2, 1, 9); mask[1, :, 6:] = 0
        pos = p.cfm.estimator._positions(mask)
        self.assertEqual(tuple(pos.shape), (2, 9, 64))
        # last valid token of the short row looks like the last token of the long row
        self.assertTrue(torch.allclose(pos[1, 5, 32:], pos[0, 8, 32:]))
        y = p.sample(torch.randn(2, 16, 9), mask, n_steps=2)
        self.assertEqual(torch.count_nonzero(y[1, :, 6:]).item(), 0)
        with tempfile.TemporaryDirectory() as d:
            p.save(Path(d) / "p.pt")
            self.assertIsNotNone(ProsodyPredictor.load(Path(d) / "p.pt").cfm.estimator.pos_proj)

    def test_weighted_loss_and_last_word_mask(self):
        from piper_train.prosody.train_predictor import last_word_mask

        # ^ _ a _ " " _ b _ c _ ? _ $   -> last word = "_ b _ c" (indices 5..8)
        pid = [1, 0, 20, 0, 3, 0, 21, 0, 22, 0, 13, 0, 2]
        m = last_word_mask(pid, 3, {13, 0, 2})
        self.assertEqual(np.nonzero(m)[0].tolist(), [5, 6, 7, 8])
        p = _pred().train()
        torch.nn.init.normal_(p.cfm.estimator.out.weight)
        x, y, mk = torch.randn(2, 16, 9), torch.randn(2, N_TARGETS, 9), torch.ones(2, 1, 9)
        torch.manual_seed(0); a = p.loss(x, y, mk)
        torch.manual_seed(0); b = p.loss(x, y, mk, weight=torch.ones(2, 1, 9))
        self.assertTrue(torch.allclose(a, b))  # unit weights = plain loss
        b.backward()

    def test_fix_voiceless(self):
        from piper_train.prosody.predictor import fix_voiceless

        pros = torch.ones(1, 3, 4)  # lf0, vfrac, energy
        out = fix_voiceless(pros, [[0, 1, 2, 3]], {0: "f", 1: "i", 2: "s", 3: "v"})
        self.assertEqual(out[0, 0].tolist(), [0.0, 1.0, 0.0, 1.0])
        self.assertTrue(torch.allclose(out[0, 1], torch.tensor([0.4, 1.0, 0.4, 1.0])))
        self.assertEqual(out[0, 2].tolist(), [1.0] * 4)

    def test_fix_voicing(self):
        from piper_train.prosody.predictor import fix_voicing

        sym = ["ð", "_", "a", "_", "t", "_", "m"]
        lf0 = torch.tensor([1.0, 2.0, 0.0, 0.0, 0.5, 0.0, 0.0])
        vf = torch.tensor([1.0, 1.0, 0.3, 0.2, 0.9, 0.1, 0.4])
        pros = torch.stack([lf0, vf, torch.zeros(7)])[None]
        out = fix_voicing(pros, [list(range(7))], dict(enumerate(sym)))[0]
        # "a" forced voiced, pitch carried from the last voiced token (no voiced after it)
        self.assertEqual(out[1, 2].item(), 1.0)
        self.assertAlmostEqual(out[0, 2].item(), 2.0)
        # pad between a and t: t is not VOICED, so it is left alone
        self.assertAlmostEqual(out[1, 3].item(), 0.2, places=5)
        # t forced unvoiced; m forced voiced
        self.assertEqual((out[0, 4].item(), out[1, 4].item() <= 0.4 + 1e-6), (0.0, True))
        self.assertEqual(out[1, 6].item(), 1.0)
        # interpolation between voiced neighbours
        pros2 = torch.tensor([[[0.0, 0.0, 2.0], [1.0, 0.2, 1.0], [0.0, 0.0, 0.0]]])
        out2 = fix_voicing(pros2, [[0, 1, 2]], {0: "ð", 1: "a", 2: "ð"})[0]
        self.assertAlmostEqual(out2[0, 1].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
