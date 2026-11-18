import unittest

import torch

from piper_train.vits.dataset import UtteranceCollate, UtteranceTensors
from piper_train.vits.models import SynthesizerTrn

KW = dict(
    n_vocab=20, spec_channels=33, segment_size=8, inter_channels=16, hidden_channels=16,
    filter_channels=32, n_heads=2, n_layers=1, kernel_size=3, p_dropout=0.0, resblock="2",
    resblock_kernel_sizes=(3,), resblock_dilation_sizes=((1, 3),), upsample_rates=(4, 4),
    upsample_initial_channel=16, upsample_kernel_sizes=(8, 8),
)


class TestProsodyConditioning(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.base = SynthesizerTrn(**KW).eval()
        self.pros = SynthesizerTrn(**KW, prosody_channels=3).eval()
        missing = self.pros.load_state_dict(self.base.state_dict(), strict=False).missing_keys
        self.assertTrue(missing and all(k.startswith("prosody_enc.") for k in missing))
        self.x = torch.randint(1, 20, (2, 7))
        self.xl = torch.LongTensor([7, 5])

    def test_zero_init_matches_pretrained(self):
        p = torch.randn(2, 3, 7)
        dur = torch.randint(1, 4, (2, 7)).float()
        torch.manual_seed(1)
        a = self.base.infer(self.x, self.xl, durations=dur, noise_scale=0.0)[0]
        torch.manual_seed(1)
        b = self.pros.infer(self.x, self.xl, durations=dur, noise_scale=0.0, prosody=p)[0]
        self.assertTrue(torch.allclose(a, b, atol=1e-5))

    def test_prosody_changes_output_once_trained(self):
        torch.nn.init.normal_(self.pros.prosody_enc[-1].weight, std=0.5)
        dur = torch.full((2, 7), 2.0)
        a = self.pros.infer(self.x, self.xl, durations=dur, noise_scale=0.0)[0]
        b = self.pros.infer(self.x, self.xl, durations=dur, noise_scale=0.0, prosody=torch.randn(2, 3, 7))[0]
        self.assertFalse(torch.allclose(a, b))

    def test_gt_durations_set_length(self):
        dur = torch.tensor([[1, 2, 3, 1, 1, 2, 2], [2, 2, 2, 2, 2, 0, 0]]).float()
        _, attn, y_mask, _ = self.pros.infer(self.x, self.xl, durations=dur)
        self.assertEqual(y_mask.sum(-1).squeeze(1).tolist(), [12, 10])

    def test_training_forward(self):
        self.pros.train()
        y = torch.randn(2, 33, 20)
        yl = torch.LongTensor([20, 16])
        out = self.pros(self.x, self.xl, y, yl, prosody=torch.randn(2, 3, 7))
        m_p, logs_p = out[6][2], out[6][3]
        (m_p.sum() + logs_p.sum()).backward()  # the KL path
        g = self.pros.prosody_enc[-1].weight.grad
        self.assertIsNotNone(g)
        self.assertGreater(g.abs().sum().item(), 0)


class TestCollateProsody(unittest.TestCase):
    def test_padding_and_sort_order(self):
        def utt(n_tok, n_frames, v):
            return UtteranceTensors(
                phoneme_ids=torch.ones(n_tok, dtype=torch.long),
                spectrogram=torch.zeros(5, n_frames),
                audio_norm=torch.zeros(1, n_frames * 4),
                prosody=torch.full((3, n_tok), float(v)),
            )
        b = UtteranceCollate(False, 16)([utt(3, 4, 1), utt(5, 9, 2)])
        self.assertEqual(tuple(b.prosody.shape), (2, 3, 5))
        self.assertTrue(torch.all(b.prosody[0] == 2))  # longest spectrogram first
        self.assertTrue(torch.all(b.prosody[1, :, :3] == 1) and torch.all(b.prosody[1, :, 3:] == 0))


if __name__ == "__main__":
    unittest.main()
