"""Per-token prosody feature matrix shared by training data and inference.

Channels (all per Piper token, BOS/pads/EOS included):
  0  log-F0, standardised with the dataset stats (0 where unvoiced)
  1  voiced fraction of the token's frames
  2  log spectral energy, standardised
  3  vocal effort: spectral tilt, log(2-5 kHz power / 50 Hz-1 kHz power),
     measured on voiced tokens and interpolated across the others
     (``voiced_effort``, when the stats say ``tilt_voiced``), standardised with
     voiced-token statistics.  Present when the prosody stats carry
     ``tilt_mean``; older extractions have three channels and keep working.
An all-zero matrix is the "no prosody given" input (also used for dropout).
"""

import numpy as np

N_CHANNELS = 3  # legacy default (no vocal-effort channel)


VOICELESS = set("ptkqʔfθsʃçxχħhɸʂɕʍɬ") | {"ts", "tʃ", "tɕ", "ʈʂ"}
VOICED = set("aeiouyɑɐɒæəɘɚɛɜɝɞɨɪʊʌɔøœɯɤʏɵɶmnŋɲɴlɫɹrɾɽʀʁjwʋɥʎʟː")


def fix_voicing_np(prosody: np.ndarray, symbols, max_vfrac: float = 0.4) -> np.ndarray:
    """Voicing by phoneme identity (numpy twin of ``predictor.fix_voicing``).

    prosody: [C, T] (lf0 standardised with 0 = unvoiced, vfrac, ...).  Voiceless
    consonants get vfrac <= max_vfrac and no pitch; vowels, sonorants and pads
    between two of them become voiced, with log-F0 interpolated linearly from
    the voiced tokens around them.  Removes voicing that flips on a voiceless
    stop or off in the middle of a vowel (heard as a blip).
    """
    p = prosody.copy()
    lf0, vf = p[0], p[1]
    n = len(symbols)
    force_on = []
    for i, s in enumerate(symbols):
        if s in VOICELESS:
            vf[i] = min(vf[i], max_vfrac)
            lf0[i] = 0.0
        elif s in VOICED or (s == "_" and 0 < i < n - 1 and symbols[i - 1] in VOICED and symbols[i + 1] in VOICED):
            if vf[i] <= 0.5:
                force_on.append(i)
    voiced = [i for i in range(n) if vf[i] > 0.5 and i not in force_on]
    if voiced and force_on:
        lf0[force_on] = np.interp(force_on, voiced, lf0[voiced])
        vf[force_on] = 1.0
    return p


def soft_pitch_ceiling(lf0: np.ndarray, vfrac: np.ndarray, ceiling: float, knee: float = 0.25) -> np.ndarray:
    """Soft-limit standardised log-F0 of voiced tokens above ``ceiling`` (a style's p95 in
    the training data): style exaggeration can otherwise push pitch beyond anything the
    speaker produced, which comes out as a frozen squeal.  Below the ceiling: unchanged."""
    out = np.asarray(lf0, np.float64).copy()
    hi = (np.asarray(vfrac) > 0.5) & (out > ceiling)
    out[hi] = ceiling + knee * np.tanh((out[hi] - ceiling) / knee)
    return out.astype(np.float32)


STRESS_MARKS = set("ˈˌ")


def cap_stress_frames(frames: np.ndarray, symbols, cap: float) -> np.ndarray:
    """Stress marks carry frames in Piper's token grid but no sound of their own; a long one
    is heard as an extra vowel ("PM" -> "P-E-M").  Cap them at ``cap`` frames and give the
    excess to the next vowel or sonorant (within 4 tokens), so the timing is kept."""
    f = np.asarray(frames, np.float32).copy()
    for i, s in enumerate(symbols):
        if s in STRESS_MARKS and f[i] > cap:
            for j in range(i + 1, min(i + 5, len(symbols))):
                if symbols[j] in VOICED:
                    f[j] += f[i] - cap
                    f[i] = cap
                    break
    return f


def voiced_effort(tilt: np.ndarray, vfrac: np.ndarray) -> np.ndarray:
    """Vocal effort lives in voiced sounds: keep the voiced tokens' tilt and
    interpolate it linearly across consonants, pads and pauses (the way pitch
    is bridged), so brightness cannot swing on tokens where it means nothing.
    An utterance without voiced tokens (whispering) keeps its mean tilt."""
    t = np.asarray(tilt, np.float64).copy()
    v = np.asarray(vfrac) > 0.5
    if not v.any():
        return np.full_like(t, t.mean() if len(t) else 0.0)
    idx = np.arange(len(t))
    t[~v] = np.interp(idx[~v], idx[v], t[v])
    return t


def n_channels(stats: dict) -> int:
    return 4 if "tilt_mean" in stats else 3


def feature_matrix(utt: dict, stats: dict) -> np.ndarray:
    """utt: one entry of prosody.pt['utts'] -> [n_tok, n_channels(stats)] float32."""
    lf0 = (utt["lf0"] - stats["lf0_mean"]) / stats["lf0_std"]
    lf0 = np.where(np.isnan(lf0), 0.0, lf0)
    en = (utt["energy"] - stats["energy_mean"]) / stats["energy_std"]
    cols = [lf0, utt["vfrac"], en]
    if n_channels(stats) == 4:
        tilt = voiced_effort(utt["tilt"], utt["vfrac"]) if stats.get("tilt_voiced") else utt["tilt"]
        cols.append((tilt - stats["tilt_mean"]) / stats["tilt_std"])
    return np.stack(cols, axis=1).astype(np.float32)
