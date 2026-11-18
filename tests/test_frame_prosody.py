"""Stage 5: frame-level interpolation of per-token prosody."""

import unittest

import torch

from piper_train.vits.models import interp_prosody_frames


class FrameProsodyTest(unittest.TestCase):
    def test_piecewise_linear_between_voiced_centres(self):
        # 3 tokens, 2 frames each; middle token unvoiced
        pros = torch.tensor([[[1.0, 0.0, 3.0], [1.0, 0.0, 1.0], [0.0, 2.0, 4.0]]])
        w = torch.tensor([[[2.0, 2.0, 2.0]]])
        f = interp_prosody_frames(pros, w, 6)[0]
        # lf0 knots at centres 1 (1.0) and 5 (3.0); frames at 0.5..5.5
        self.assertTrue(torch.allclose(f[0], torch.tensor([1.0, 1.25, 1.75, 2.25, 2.75, 3.0])))
        # energy uses every token: knots at 1, 3, 5 with values 0, 2, 4
        self.assertTrue(torch.allclose(f[2], torch.tensor([0.0, 0.5, 1.5, 2.5, 3.5, 4.0])))

    def test_zero_duration_tokens_are_skipped(self):
        pros = torch.tensor([[[1.0, 9.0, 3.0], [1.0, 1.0, 1.0]]])
        w = torch.tensor([[[2.0, 0.0, 2.0]]])
        f = interp_prosody_frames(pros, w, 4)[0]
        self.assertTrue(torch.allclose(f[0], torch.tensor([1.0, 1.5, 2.5, 3.0])))


if __name__ == "__main__":
    unittest.main()
