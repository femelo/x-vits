"""Per-token prosody feature matrix shared by training data and inference.

Channels (all per Piper token, BOS/pads/EOS included):
  0  log-F0, standardised with the dataset stats (0 where unvoiced)
  1  voiced fraction of the token's frames
  2  log spectral energy, standardised
  3  vocal effort: spectral tilt, log(2-5 kHz power / 50 Hz-1 kHz power),
     standardised.  Present when the prosody stats carry ``tilt_mean``;
     older extractions have three channels and keep working.
An all-zero matrix is the "no prosody given" input (also used for dropout).
"""

import numpy as np

N_CHANNELS = 3  # legacy default (no vocal-effort channel)


def n_channels(stats: dict) -> int:
    return 4 if "tilt_mean" in stats else 3


def feature_matrix(utt: dict, stats: dict) -> np.ndarray:
    """utt: one entry of prosody.pt['utts'] -> [n_tok, n_channels(stats)] float32."""
    lf0 = (utt["lf0"] - stats["lf0_mean"]) / stats["lf0_std"]
    lf0 = np.where(np.isnan(lf0), 0.0, lf0)
    en = (utt["energy"] - stats["energy_mean"]) / stats["energy_std"]
    cols = [lf0, utt["vfrac"], en]
    if n_channels(stats) == 4:
        cols.append((utt["tilt"] - stats["tilt_mean"]) / stats["tilt_std"])
    return np.stack(cols, axis=1).astype(np.float32)
