"""Per-token prosody feature matrix shared by training data and inference.

Channels (all per Piper token, BOS/pads/EOS included):
  0  log-F0, standardised with the dataset stats (0 where unvoiced)
  1  voiced fraction of the token's frames
  2  log spectral energy, standardised
An all-zero matrix is the "no prosody given" input (also used for dropout).
"""

import numpy as np

N_CHANNELS = 3


def feature_matrix(utt: dict, stats: dict) -> np.ndarray:
    """utt: one entry of prosody.pt['utts'] -> [n_tok, 3] float32."""
    lf0 = (utt["lf0"] - stats["lf0_mean"]) / stats["lf0_std"]
    lf0 = np.where(np.isnan(lf0), 0.0, lf0)
    en = (utt["energy"] - stats["energy_mean"]) / stats["energy_std"]
    return np.stack([lf0, utt["vfrac"], en], axis=1).astype(np.float32)
