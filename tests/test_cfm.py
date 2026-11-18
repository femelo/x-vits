"""CPU unit tests for the OT-CFM prior and the mu_z / MAS change.

Run (no pytest needed):  python -m unittest discover -s tests -t .
"""

import math
import unittest
import warnings

import torch

warnings.filterwarnings("ignore")

from piper_train.vits import monotonic_align
from piper_train.vits.cfm import CFMEstimator, OTCFM, sinusoidal_embedding
from piper_train.vits.losses import kl_std_normal_loss
from piper_train.vits.models import SynthesizerTrn

N_VOCAB, SPEC, INTER, HID = 50, 513, 192, 192


def make_model(**kw) -> SynthesizerTrn:
    return SynthesizerTrn(
        n_vocab=N_VOCAB,
        spec_channels=SPEC,
        segment_size=32,
        inter_channels=INTER,
        hidden_channels=HID,
        filter_channels=768,
        n_heads=2,
        n_layers=2,
        kernel_size=3,
        p_dropout=0.1,
        resblock="2",
        resblock_kernel_sizes=(3, 5, 7),
        resblock_dilation_sizes=((1, 2), (2, 6), (3, 12)),
        upsample_rates=(8, 8, 4),
        upsample_initial_channel=64,
        upsample_kernel_sizes=(16, 16, 8),
        **kw,
    )


def make_batch(b=2, t_s=12, t_t=60):
    x = torch.randint(1, N_VOCAB, (b, t_s))
    xl = torch.tensor([t_s, t_s - 3][:b])
    y = torch.randn(b, SPEC, t_t)
    yl = torch.tensor([t_t, t_t - 15][:b])
    return x, xl, y, yl


class TestEstimator(unittest.TestCase):
    def test_time_embedding_shape(self):
        self.assertEqual(sinusoidal_embedding(torch.rand(3), 64).shape, (3, 64))
        self.assertEqual(sinusoidal_embedding(torch.rand(3), 63).shape, (3, 63))

    def test_zero_init_and_mask(self):
        est = CFMEstimator(16, hidden_channels=32, n_layers=3, gin_channels=8)
        x, mu = torch.randn(2, 16, 20), torch.randn(2, 16, 20)
        mask = torch.ones(2, 1, 20)
        mask[1, :, 12:] = 0
        v = est(x, mask, mu, torch.rand(2), g=torch.randn(2, 8, 1))
        self.assertEqual(v.shape, x.shape)
        self.assertTrue(torch.all(v == 0), "zero-initialised output projection")
        est.proj.weight.data.normal_()
        v = est(x, mask, mu, torch.rand(2), g=torch.randn(2, 8, 1))
        self.assertTrue(torch.all(v[1, :, 12:] == 0), "padding must be masked")

    def test_time_and_speaker_conditioning_matter(self):
        est = CFMEstimator(16, hidden_channels=32, n_layers=3, gin_channels=8)
        est.proj.weight.data.normal_()
        x, mu, mask = torch.randn(1, 16, 20), torch.randn(1, 16, 20), torch.ones(1, 1, 20)
        g = torch.randn(1, 8, 1)
        a = est(x, mask, mu, torch.tensor([0.1]), g=g)
        b = est(x, mask, mu, torch.tensor([0.9]), g=g)
        c = est(x, mask, mu, torch.tensor([0.1]), g=g + 1)
        self.assertFalse(torch.allclose(a, b))
        self.assertFalse(torch.allclose(a, c))


class TestOTCFM(unittest.TestCase):
    def test_path_and_target(self):
        """At t=0 x_t is the noise, at t=1 it is the data (up to sigma_min)."""
        cfm = OTCFM(CFMEstimator(4, 16, 2), sigma_min=1e-4)
        x1 = torch.randn(2, 4, 10)
        mask = torch.ones(2, 1, 10)
        torch.manual_seed(0)
        _, x_t1 = cfm.compute_loss(x1, mask, x1, t=torch.ones(2))
        self.assertTrue(torch.allclose(x_t1, x1, atol=1e-3))
        torch.manual_seed(0)
        _, x_t0 = cfm.compute_loss(x1, mask, x1, t=torch.zeros(2))
        torch.manual_seed(0)
        self.assertTrue(torch.allclose(x_t0, torch.randn_like(x1)))

    def test_loss_ignores_padding(self):
        cfm = OTCFM(CFMEstimator(4, 16, 2))
        cfm.estimator.proj.weight.data.normal_()
        x1 = torch.randn(1, 4, 10)
        mask = torch.ones(1, 1, 10)
        mask[..., 6:] = 0
        t = torch.tensor([0.5])
        torch.manual_seed(1)
        l1, _ = cfm.compute_loss(x1, mask, x1, t=t)
        x1b = x1.clone()
        x1b[..., 6:] = 100.0
        torch.manual_seed(1)
        l2, _ = cfm.compute_loss(x1b, mask, x1, t=t)
        # padded region changes x_t only there; valid-frame loss must agree up to
        # receptive-field leakage, which the masks in WN block.
        self.assertAlmostEqual(l1.item(), l2.item(), places=4)

    def test_learns_toy_distribution(self):
        """Estimator fits a conditional target: z = 2*mu + 1 (deterministic given mu)."""
        torch.manual_seed(0)
        cfm = OTCFM(CFMEstimator(4, 64, 4, kernel_size=3, dilation_rate=2))
        opt = torch.optim.Adam(cfm.parameters(), lr=3e-3)
        mask = torch.ones(8, 1, 16)
        first = last = None
        for step in range(150):
            mu = torch.randn(8, 4, 16)
            loss, _ = cfm.compute_loss(2 * mu + 1, mask, mu)
            opt.zero_grad()
            loss.backward()
            opt.step()
            first = first if first is not None else loss.item()
            last = loss.item()
        self.assertLess(last, 0.6 * first)
        mu = torch.randn(4, 4, 16)
        z = cfm.sample(mu, torch.ones(4, 1, 16), n_steps=8, temperature=0.0)
        err = (z - (2 * mu + 1)).abs().mean().item()
        self.assertLess(err, 0.5)

    def test_sample_shape_mask_and_determinism(self):
        cfm = OTCFM(CFMEstimator(4, 16, 2, gin_channels=3))
        cfm.estimator.proj.weight.data.normal_()
        mu = torch.randn(2, 4, 10)
        mask = torch.ones(2, 1, 10)
        mask[1, :, 7:] = 0
        g = torch.randn(2, 3, 1)
        noise = torch.randn_like(mu)
        a = cfm.sample(mu, mask, g=g, n_steps=3, noise=noise)
        b = cfm.sample(mu, mask, g=g, n_steps=3, noise=noise)
        self.assertEqual(a.shape, mu.shape)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.all(a[1, :, 7:] == 0))


class TestMAS(unittest.TestCase):
    def test_gaussian_matches_legacy_formula(self):
        z, mu, logs = torch.randn(2, 8, 30), torch.randn(2, 8, 10), torch.randn(2, 8, 10) * 0.3
        got = SynthesizerTrn._neg_cent_gaussian(z, mu, logs)
        s = torch.exp(-2 * logs)
        c1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs, [1], keepdim=True)
        c2 = torch.matmul(-0.5 * (z**2).transpose(1, 2), s)
        c3 = torch.matmul(z.transpose(1, 2), mu * s)
        c4 = torch.sum(-0.5 * (mu**2) * s, [1], keepdim=True)
        self.assertTrue(torch.allclose(got, c1 + c2 + c3 + c4, atol=1e-4))

    def test_unit_variance_equals_negative_sq_distance(self):
        z, mu = torch.randn(1, 8, 30), torch.randn(1, 8, 10)
        got = SynthesizerTrn._neg_cent_gaussian(z, mu)
        ref = -0.5 * ((z.unsqueeze(3) - mu.unsqueeze(2)) ** 2).sum(1)  # [b, t_t, t_s]
        # equal up to a per-frame constant (-0.5 |z_t|^2) that MAS ignores
        const = -0.5 * (z**2).sum(1).unsqueeze(2)
        self.assertTrue(torch.allclose(got + const, ref, atol=1e-4))

    def test_mas_recovers_known_alignment(self):
        torch.manual_seed(0)
        durs = [3, 5, 2, 6, 4]
        mu = torch.randn(1, 16, len(durs)) * 3
        z = torch.cat([mu[:, :, i : i + 1].expand(-1, -1, d) for i, d in enumerate(durs)], 2)
        z = z + 0.1 * torch.randn_like(z)
        t_t, t_s = z.size(2), mu.size(2)
        neg = SynthesizerTrn._neg_cent_gaussian(z, mu)
        path = monotonic_align.maximum_path(neg, torch.ones(1, t_t, t_s))
        self.assertEqual(path.sum(1).squeeze().long().tolist(), durs)


class TestSynthesizer(unittest.TestCase):
    def test_vits_mode_unchanged(self):
        m = make_model()
        out = m(*make_batch())
        self.assertEqual(len(out), 8)
        self.assertIsNone(out[-1])
        z, z_p, m_p, logs_p, m_q, logs_q = out[6]
        self.assertIsNotNone(z_p)
        self.assertIsNotNone(logs_p)
        self.assertFalse(hasattr(m, "cfm"))

    def test_cfm_forward_and_backward(self):
        m = make_model(use_cfm=True, cfm_hidden_channels=32, cfm_n_layers=3)
        out = m(*make_batch())
        aux = out[-1]
        self.assertEqual(set(aux), {"loss_cfm", "loss_prior"})
        self.assertTrue(all(torch.isfinite(v) for v in aux.values()))
        self.assertEqual(out[2].shape[2:], (60, 12))  # attn [b,1,t_t,t_s]
        # alignment rows sum to 1 inside the valid region
        attn = out[2].squeeze(1)
        self.assertTrue(torch.all(attn.sum(2)[0] == 1))

    def test_cfm_loss_does_not_touch_posterior_or_frozen_parts(self):
        m = make_model(use_cfm=True, cfm_hidden_channels=32, cfm_n_layers=3)
        m.cfm.estimator.proj.weight.data.normal_(0, 0.1)  # nonzero grads
        aux = m(*make_batch())[-1]
        (aux["loss_cfm"] + aux["loss_prior"]).backward()
        enc_q_grads = [p.grad for p in m.enc_q.parameters()]
        self.assertTrue(all(g is None for g in enc_q_grads), "z target must be detached")
        self.assertTrue(all(p.grad is None for p in m.flow.parameters()))
        self.assertIsNotNone(m.mu_proj.weight.grad)
        self.assertIsNotNone(m.cfm.estimator.pre.weight.grad)
        self.assertFalse(any(p.requires_grad for p in m.flow.parameters()))
        self.assertFalse(any(p.requires_grad for p in m.enc_p.proj.parameters()))

    def test_flow_mas_warmup_path(self):
        m = make_model(use_cfm=True, cfm_hidden_channels=32, cfm_n_layers=3)
        a = m(*make_batch(), use_flow_mas=True)
        self.assertTrue(all(torch.isfinite(v) for v in a[-1].values()))

    def test_infer_cfm(self):
        m = make_model(use_cfm=True, cfm_hidden_channels=32, cfm_n_layers=3).eval()
        x, xl, _, _ = make_batch()
        with torch.no_grad():
            o, attn, y_mask, (z, _, mu, _) = m.infer(x, xl, n_steps=2)
        self.assertEqual(o.size(0), 2)
        self.assertEqual(z.shape, mu.shape)
        self.assertEqual(o.size(2), z.size(2) * 256)
        self.assertTrue(torch.isfinite(o).all())
        with torch.no_grad():
            o1 = m.infer(x, xl, noise_scale=0.0)[0]
            o2 = m.infer(x, xl, noise_scale=0.0)[0]
        # temperature 0 + deterministic dp-free durations (SDP noise differs), so
        # only check the vocoder path is finite here
        self.assertTrue(torch.isfinite(o1).all() and torch.isfinite(o2).all())

    def test_checkpoint_compat_with_vits(self):
        """A VITS state_dict loads into a CFM model; only the new modules are missing."""
        vits = make_model()
        cfm = make_model(use_cfm=True)
        res = cfm.load_state_dict(vits.state_dict(), strict=False)
        self.assertEqual(res.unexpected_keys, [])
        self.assertTrue(res.missing_keys)
        self.assertTrue(all(k.startswith(("mu_proj.", "cfm.")) for k in res.missing_keys))
        for k, v in vits.state_dict().items():
            self.assertTrue(torch.equal(cfm.state_dict()[k], v), k)

    def test_voice_conversion_rejected_in_cfm_mode(self):
        m = make_model(use_cfm=True, n_speakers=2, gin_channels=16)
        with self.assertRaises(NotImplementedError):
            m.voice_conversion(None, None, None, None)

    def test_multispeaker_cfm(self):
        m = make_model(use_cfm=True, n_speakers=3, gin_channels=16, cfm_hidden_channels=32, cfm_n_layers=3)
        out = m(*make_batch(), sid=torch.tensor([0, 2]))
        self.assertTrue(torch.isfinite(out[-1]["loss_cfm"]))
        with torch.no_grad():
            m.eval().infer(make_batch()[0], make_batch()[1], sid=torch.tensor([0, 1]), n_steps=2)


class TestLosses(unittest.TestCase):
    def test_kl_std_normal(self):
        mask = torch.ones(1, 1, 5)
        zero = torch.zeros(1, 4, 5)
        self.assertAlmostEqual(kl_std_normal_loss(zero, zero, mask).item(), 0.0, places=6)
        # closed form for N(1, 1) vs N(0, 1): 0.5 per element
        self.assertAlmostEqual(kl_std_normal_loss(zero + 1, zero, mask).item(), 0.5, places=6)
        mask[..., 3:] = 0
        big = torch.zeros(1, 4, 5)
        big[..., 3:] = 100
        self.assertAlmostEqual(kl_std_normal_loss(big, zero, mask).item(), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
