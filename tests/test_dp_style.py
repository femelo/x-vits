"""Style conditioning hook of the stochastic duration predictor."""

import unittest

import torch

from piper_train.vits.models import StochasticDurationPredictor


class DpStyleTest(unittest.TestCase):
    def test_h_add_zero_is_identity_and_nonzero_changes_output(self):
        torch.manual_seed(0)
        dp = StochasticDurationPredictor(16, 16, 3, 0.0, 4).eval()
        with torch.no_grad():  # flows start as identities, which would ignore x
            for q in dp.parameters():
                q.add_(0.05 * torch.randn_like(q))
        x, m = torch.randn(1, 16, 7), torch.ones(1, 1, 7)
        w = torch.randint(1, 6, (1, 1, 7)).float()
        torch.manual_seed(1); a = dp(x, m, w)
        torch.manual_seed(1); b = dp(x, m, w, h_add=torch.zeros(1, 16, 1))
        self.assertTrue(torch.allclose(a, b))
        torch.manual_seed(1); c = dp(x, m, w, h_add=torch.randn(1, 16, 1, generator=torch.Generator().manual_seed(5)))
        self.assertFalse(torch.allclose(a, c))
        torch.manual_seed(2); r0 = dp(x, m, reverse=True, noise_scale=0.0)
        torch.manual_seed(2); r1 = dp(x, m, reverse=True, noise_scale=0.0, h_add=torch.randn(1, 16, 1, generator=torch.Generator().manual_seed(5)))
        self.assertFalse(torch.allclose(r0, r1))


if __name__ == "__main__":
    unittest.main()
