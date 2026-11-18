"""Stage 4: emphasis markup, token features and SDP conditioning."""

import tempfile
import unittest
from pathlib import Path

import torch

from piper_train.prosody.emphasis import parse_marked
from piper_train.prosody.predictor import N_TARGETS, ProsodyPredictor
from piper_train.prosody.train_dp_style import DpCond

STATS = {"lf0_mean": 5.3, "lf0_std": 0.22, "energy_mean": 2.6, "energy_std": 1.2,
         "logdur_mean": 1.17, "logdur_std": 0.47}


class EmphasisTest(unittest.TestCase):
    def test_parse_marked(self):
        self.assertEqual(parse_marked("You *want* the shark?"),
                         ("You want the shark?", [False, True, False, False]))
        self.assertEqual(parse_marked("I am really, *really* happy."),
                         ("I am really, really happy.", [False, False, False, True, False]))
        self.assertEqual(parse_marked("in the *World Cup*!"),
                         ("in the World Cup!", [False, False, True, True]))
        self.assertEqual(parse_marked("No marks here."), ("No marks here.", [False] * 3))

    def test_predictor_token_feats(self):
        p = ProsodyPredictor(STATS, ["a"], 1, text_channels=16, hidden=32, n_wn_layers=2,
                             n_attn_layers=1, n_heads=2).eval()
        torch.nn.init.normal_(p.cfm.estimator.out.weight)
        x, m = torch.randn(1, 16, 9), torch.ones(1, 1, 9)
        f = torch.zeros(1, 1, 9); f[0, 0, 3:6] = 1
        torch.manual_seed(0); a = p.sample(x, m, n_steps=2)
        torch.manual_seed(0); b = p.sample(x, m, n_steps=2, feats=torch.zeros(1, 1, 9))
        torch.manual_seed(0); c = p.sample(x, m, n_steps=2, feats=f)
        self.assertTrue(torch.allclose(a, b))  # missing feats == zeros
        self.assertFalse(torch.allclose(a, c))
        p.train().loss(x, torch.randn(1, N_TARGETS, 9), m, feats=f).backward()
        with tempfile.TemporaryDirectory() as d:
            p.save(Path(d) / "p.pt")
            self.assertEqual(ProsodyPredictor.load(Path(d) / "p.pt").token_feats, 1)

    def test_dp_cond(self):
        c = DpCond(3, 8, emphasis=True)
        g = torch.LongTensor([1])
        self.assertEqual(tuple(c(g).shape), (1, 8, 1))
        self.assertEqual(tuple(c(g, torch.ones(1, 1, 5)).shape), (1, 8, 5))
        self.assertEqual(torch.count_nonzero(c(g, torch.ones(1, 1, 5))).item(), 0)  # zero init

    def test_feature_guidance_and_multichannel_dp_cond(self):
        p = ProsodyPredictor(STATS, ["a"], 3, text_channels=16, hidden=32, n_wn_layers=2,
                             n_attn_layers=1, n_heads=2).eval()
        torch.nn.init.normal_(p.cfm.estimator.out.weight)
        x, m = torch.randn(1, 16, 9), torch.ones(1, 1, 9)
        f = torch.zeros(1, 3, 9); f[0, :, 3:6] = 1
        g = torch.LongTensor([1])
        torch.manual_seed(0); a = p.sample(x, m, n_steps=2, style=g, feats=f)
        torch.manual_seed(0); b = p.sample(x, m, n_steps=2, style=g, feats=f, feats_guidance=1.0)
        torch.manual_seed(0); c = p.sample(x, m, n_steps=2, style=g, feats=f, feats_guidance=2.0)
        self.assertTrue(torch.allclose(a, b))
        self.assertFalse(torch.allclose(a, c))
        d = DpCond(3, 8, emphasis=3)
        self.assertEqual(tuple(d(g, torch.ones(1, 3, 5)).shape), (1, 8, 5))

    def test_template(self):
        import numpy as np
        from piper_train.prosody.emphasis import apply_template, word_runs

        rel = word_runs(np.array([0, 1, 1, 1, 0, 1], np.float32))
        self.assertTrue(np.isnan(rel[0]) and np.isnan(rel[4]))
        self.assertEqual(rel[1:4].tolist(), [0.0, 0.5, 1.0])
        self.assertEqual(rel[5], 0.5)
        tpl = {"bins": 2, "pitch_st": [2.0, 0.0], "dur_mult": [1.5, 1.0]}
        p, d = apply_template(rel, tpl, scale=1.0)
        self.assertEqual((p[0], d[0]), (0.0, 1.0))
        self.assertAlmostEqual(float(p[1]), 2.0); self.assertAlmostEqual(float(d[1]), 1.5)
        p2, d2 = apply_template(rel, tpl, scale=2.0)
        self.assertAlmostEqual(float(p2[1]), 4.0); self.assertAlmostEqual(float(d2[1]), 2.25)

    def test_tonic_offsets_and_param_template(self):
        import numpy as np
        from piper_train.prosody.emphasis import apply_param_template, tonic_offsets

        # word 0: d ˈ a t ; word 1 (unflagged): o
        sym = {0: "d", 1: "ˈ", 2: "a", 3: "t", 4: " ", 5: "o"}
        ids, tok_w = [0, 1, 2, 3, 4, 5], [0, 0, 0, 0, -1, 1]
        off = tonic_offsets(ids, tok_w, sym, flags=[True, False])
        self.assertEqual(off[:4].tolist(), [-2.0, -1.0, 0.0, 1.0])
        self.assertTrue(np.isnan(off[4]) and np.isnan(off[5]))
        tpl = {"pitch": {"A": 2.0, "mu": 0.0, "s_l": 1.0, "s_r": 1.0, "R": 0.0, "k_r": 5.0, "tau": 1.0},
               "dur": {"C": 0.0, "B": float(np.log(1.5)), "mu": 0.0, "s_l": 1.0, "s_r": 1.0},
               "styles": {"x": {"pitch": {"A": 4.0, "mu": 0.0, "s_l": 1.0, "s_r": 1.0, "R": 0.0, "k_r": 5.0, "tau": 1.0},
                                "dur": {"C": 0.0, "B": 0.0, "mu": 0.0, "s_l": 1.0, "s_r": 1.0}}}}
        p, d = apply_param_template(off, tpl)
        self.assertAlmostEqual(float(p[2]), 2.0, places=5)  # peak at the stressed vowel
        self.assertAlmostEqual(float(d[2]), 1.5, places=5)
        self.assertEqual((float(p[5]), float(d[5])), (0.0, 1.0))  # not emphasised
        p2, _ = apply_param_template(off, tpl, style="x")
        self.assertAlmostEqual(float(p2[2]), 4.0, places=5)
        p3, _ = apply_param_template(off, tpl, style="unknown")  # falls back to pooled
        self.assertAlmostEqual(float(p3[2]), 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
