"""Stage 3: flow-matching prosody predictor.

Generates, per Piper token, jointly-distributed values from the frozen text
encoder output ``x`` (``enc_p`` hidden states, [b, 192, t]):

  0  log(1 + frames), standardised      (duration; replaces the SDP)
  1  log-F0, standardised (0 unvoiced)  \
  2  voiced fraction                     |  the stage 2 prosody input
  3  log energy, standardised            |  (see features.py)
  4  vocal effort (spectral tilt)       /   only with a tilt-aware extraction

Sampling is an OT-CFM ODE from Gaussian noise, so prosody is a distribution
(the same sentence can be said several ways) rather than a regression mean,
which is what flattened pitch in the conditioned model's no-input path.

The estimator mixes local context (dilated WaveNet stack) with sentence-level
context (a few self-attention layers), since phrase-final rises, list
intonation and emphasis depend on the whole utterance.
"""

import math
from typing import Optional

import numpy as np
import torch
from torch import nn

from ..vits import modules
from ..vits.cfm import OTCFM, sinusoidal_embedding
from .features import feature_matrix, n_channels

N_TARGETS = 4  # legacy default; a predictor's own count is ``n_targets``


class ProsodyEstimator(nn.Module):
    def __init__(
        self,
        text_channels: int = 192,
        hidden: int = 256,
        n_wn_layers: int = 8,
        n_attn_layers: int = 3,
        n_heads: int = 4,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        n_styles: int = 0,
        pos_features: bool = False,
        n_targets: int = N_TARGETS,
    ):
        super().__init__()
        self.hidden = hidden
        self.n_targets = n_targets
        # Optional explicit position: distance (tokens) to the sentence start and
        # end, so phrase-final behaviour (question rises, final lengthening) does
        # not have to be inferred from the text encoder alone.
        self.pos_proj = nn.Linear(2 * 32, hidden) if pos_features else None
        # Optional utterance-level style label (index 0 = "unspecified"), added
        # to the time embedding so it reaches every WaveNet and attention layer.
        self.style = nn.Embedding(n_styles + 1, hidden) if n_styles > 0 else None
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.in_proj = nn.Conv1d(n_targets, hidden, 1)
        self.cond_proj = nn.Conv1d(text_channels, hidden, 1)
        self.wn = modules.WN(
            hidden, kernel_size, 2, n_wn_layers, gin_channels=hidden, p_dropout=p_dropout
        )
        self.attn = nn.ModuleList(
            nn.TransformerEncoderLayer(
                hidden, n_heads, 4 * hidden, p_dropout, batch_first=True, norm_first=True
            )
            for _ in range(n_attn_layers)
        )
        self.out = nn.Conv1d(hidden, n_targets, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def _positions(mask):
        m = mask.squeeze(1)  # [b, T]
        idx = torch.cumsum(m, dim=1) - 1
        d_start = idx.clamp_min(0)
        d_end = (m.sum(1, keepdim=True) - 1 - idx).clamp_min(0)
        return torch.cat([_log_distance_embedding(d_start), _log_distance_embedding(d_end)], -1)

    def forward(self, x_t, mask, cond, t, g=None):
        """x_t: [b, 4, T]; mask: [b, 1, T]; cond: [b, text_ch, T]; t: [b];
        g: style index [b] (long, 0 = unspecified) or None"""
        temb = self.time_mlp(sinusoidal_embedding(t, self.hidden))  # [b, h]
        if self.style is not None:
            if g is None:
                g = torch.zeros(t.size(0), dtype=torch.long, device=t.device)
            temb = temb + self.style(g)
        h = self.in_proj(x_t) + self.cond_proj(cond)
        if self.pos_proj is not None:
            h = h + self.pos_proj(self._positions(mask)).transpose(1, 2)
        h = h * mask
        h = self.wn(h, mask, g=temb.unsqueeze(-1))
        pad = mask.squeeze(1) < 0.5  # [b, T] True = padding
        h = h.transpose(1, 2) + temb.unsqueeze(1)
        for layer in self.attn:
            h = layer(h, src_key_padding_mask=pad)
        return self.out(h.transpose(1, 2)) * mask


def _log_distance_embedding(d: torch.Tensor, dim: int = 32) -> torch.Tensor:
    """d: [b, T] distances >= 0 -> [b, T, dim] sinusoids of log(1 + d)."""
    freqs = torch.exp(torch.linspace(0, math.log(64.0), dim // 2, device=d.device))
    a = torch.log1p(d).unsqueeze(-1) * freqs
    return torch.cat([a.sin(), a.cos()], dim=-1)


class _Guided(nn.Module):
    """Classifier-free guidance: v_uncond + w * (v_style - v_uncond)."""

    def __init__(self, est, w):
        super().__init__()
        self.est, self.w = est, w

    def forward(self, x_t, mask, cond, t, g=None):
        v_c = self.est(x_t, mask, cond, t, g=g)
        v_u = self.est(x_t, mask, cond, t, g=torch.zeros_like(g))
        return v_u + self.w * (v_c - v_u)


class _MultiGuided(nn.Module):
    """Classifier-free guidance on style and/or per-token features:
    v = v_full + (w_s - 1)(v_full - v_nostyle) + (w_e - 1)(v_full - v_nofeats)."""

    def __init__(self, est, w_style, w_feats, cond_nofeats):
        super().__init__()
        self.est, self.ws, self.we, self.cond0 = est, w_style, w_feats, cond_nofeats

    def forward(self, x_t, mask, cond, t, g=None):
        v = self.est(x_t, mask, cond, t, g=g)
        out = v
        if g is not None and bool((torch.as_tensor(self.ws) != 1.0).any()):
            out = out + (self.ws - 1.0) * (v - self.est(x_t, mask, cond, t, g=torch.zeros_like(g)))
        if self.we != 1.0:
            out = out + (self.we - 1.0) * (v - self.est(x_t, mask, self.cond0, t, g=g))
        return out


class ProsodyPredictor(nn.Module):
    """OT-CFM over the per-token targets (duration + prosody channels).  ``stats`` holds the dataset
    normalisation (prosody.pt stats plus logdur_mean/std).  ``styles`` is an
    optional list of utterance style labels (e.g. Expresso's happy/sad/...);
    without it the predictor is unconditional, as before."""

    def __init__(self, stats: dict, styles=None, token_feats: int = 0, **estimator_kw):
        super().__init__()
        self.stats = dict(stats)
        self.styles = list(styles or [])
        # extra per-token inputs (stage 4: word emphasis), concatenated to the
        # text encoder output; absent features are passed as zeros
        self.token_feats = token_feats
        self.kw = dict(estimator_kw)
        est_kw = dict(estimator_kw)
        est_kw["text_channels"] = est_kw.get("text_channels", 192) + token_feats
        self.n_targets = 1 + n_channels(self.stats)  # duration + the prosody channels
        self.cfm = OTCFM(ProsodyEstimator(n_styles=len(self.styles), n_targets=self.n_targets, **est_kw))

    def _cond(self, x_text, feats):
        if self.token_feats == 0:
            return x_text
        if feats is None:
            feats = x_text.new_zeros(x_text.size(0), self.token_feats, x_text.size(2))
        return torch.cat([x_text, feats.to(x_text.dtype)], dim=1)

    def style_index(self, name: Optional[str]) -> int:
        """Label -> embedding index (0 = unspecified / unknown label)."""
        if not name or name not in self.styles:
            if name and self.styles:
                raise ValueError(f"unknown style {name!r}; known: {self.styles}")
            return 0
        return 1 + self.styles.index(name)

    # ---- target encoding -------------------------------------------------
    def encode_targets(self, utt: dict) -> np.ndarray:
        """prosody.pt utterance -> [n_tok, n_targets] float32 (standardised)."""
        s = self.stats
        ld = (np.log1p(utt["dur"].astype(np.float32)) - s["logdur_mean"]) / s["logdur_std"]
        return np.concatenate([ld[:, None], feature_matrix(utt, s)], axis=1).astype(np.float32)

    def decode(self, y: torch.Tensor, mask: torch.Tensor):
        """y: [b, n_targets, T] -> (durations [b, T] frames >= 1, prosody [b, n_targets - 1, T])."""
        s = self.stats
        logdur = y[:, 0] * s["logdur_std"] + s["logdur_mean"]
        dur = torch.clamp_min(torch.round(torch.expm1(logdur)), 1.0) * mask.squeeze(1)
        vfrac = y[:, 2:3].clamp(0.0, 1.0)
        lf0 = y[:, 1:2] * (vfrac > 0.5)  # unvoiced tokens: "no pitch" (0)
        rest = y[:, 3:]  # energy (, vocal effort)
        pros = torch.cat([lf0, vfrac, rest], dim=1) * mask
        return dur, self.bridge_effort(pros, mask)

    def bridge_effort(self, pros, mask):
        """Effort is defined on voiced tokens: re-derive it elsewhere from the voiced
        neighbours (as in training, see features.voiced_effort).  Call again after
        anything that changes the voicing (e.g. fix_voicing)."""
        if pros.size(1) < 4 or not self.stats.get("tilt_voiced"):
            return pros
        from .features import voiced_effort

        pros = pros.clone()
        for b in range(pros.size(0)):
            n = int(mask[b].sum())
            e = voiced_effort(pros[b, 3, :n].double().cpu().numpy(), pros[b, 1, :n].double().cpu().numpy())
            pros[b, 3, :n] = torch.from_numpy(e).to(pros)
        return pros

    # ---- training / sampling ---------------------------------------------
    def loss(self, x_text, y, mask, style=None, weight=None, feats=None):
        """weight: optional per-token loss weight [b, 1, T] (e.g. to stress the
        last word of questions); None = the plain OT-CFM loss.  feats: optional
        per-token features [b, token_feats, T]."""
        x_text = self._cond(x_text, feats)
        if weight is None:
            loss, _ = self.cfm.compute_loss(y, mask, x_text, g=style)
            return loss
        cfm, b = self.cfm, y.size(0)
        t = torch.rand(b, device=y.device, dtype=y.dtype).view(b, 1, 1)
        x0 = torch.randn_like(y)
        x_t = (1 - (1 - cfm.sigma_min) * t) * x0 + t * y
        u = y - (1 - cfm.sigma_min) * x0
        v = cfm.estimator(x_t, mask, x_text, t.view(b), g=style)
        w = weight * mask
        return torch.sum(((v - u) ** 2) * w) / (torch.sum(w) * y.size(1))

    @torch.no_grad()
    def sample(self, x_text, mask, n_steps: int = 16, temperature: float = 1.0,
               solver="midpoint", style=None, guidance: float = 1.0, feats=None,
               feats_guidance: float = 1.0):
        """style: index tensor [b] (see style_index) or None; guidance > 1
        exaggerates the style relative to the unconditional prediction.
        feats: per-token features [b, token_feats, T] (None = zeros);
        feats_guidance > 1 exaggerates their effect (e.g. word emphasis)."""
        cond0 = self._cond(x_text, None)
        x_text = self._cond(x_text, feats)
        if feats is not None and self.token_feats and feats_guidance != 1.0:
            w = torch.as_tensor(guidance, dtype=x_text.dtype, device=x_text.device)
            w = w.view(1, -1, 1) if w.dim() else w
            noise = torch.randn(x_text.size(0), self.n_targets, x_text.size(2), device=x_text.device, dtype=x_text.dtype)
            cfm = OTCFM(_MultiGuided(self.cfm.estimator, w if style is not None else 1.0, feats_guidance, cond0))
            return cfm.sample(x_text, mask, n_steps=n_steps, temperature=temperature, noise=noise,
                              solver=solver, g=style)
        noise = torch.randn(
            x_text.size(0), self.n_targets, x_text.size(2), device=x_text.device, dtype=x_text.dtype
        )
        cfm = self.cfm
        w = torch.as_tensor(guidance, dtype=x_text.dtype, device=x_text.device)
        if style is not None and bool((w != 1.0).any()):
            # scalar, or per channel [1, 4, 1] (e.g. exaggerate pitch, not timing)
            cfm = OTCFM(_Guided(self.cfm.estimator, w.view(1, -1, 1) if w.dim() else w))
        return cfm.sample(
            x_text, mask, n_steps=n_steps, temperature=temperature, noise=noise,
            solver=solver, g=style,
        )

    def save(self, path, extra: Optional[dict] = None):
        torch.save(
            {"state_dict": self.state_dict(), "stats": self.stats, "kw": self.kw,
             "styles": self.styles, "token_feats": self.token_feats, **(extra or {})},
            path,
        )

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        m = cls(ck["stats"], ck.get("styles"), ck.get("token_feats", 0), **ck["kw"])
        m.load_state_dict(ck["state_dict"])
        return m.to(device).eval()


PAUSE_SYMBOLS = set(" !\"#$'(),-.:;?^_")


def shape_durations(dur, phoneme_ids, id_to_symbol, max_frames=None, final_stretch=1.0):
    """Post-process sampled durations [b, T] (frames) in place-safe fashion.

    max_frames: cap on tokens that cannot carry a pause (phonemes and the
      inter-phoneme pads inside words), against isolated over-stretched sounds.
      Pads next to a space or punctuation and the punctuation itself are left
      alone, since that is where pauses live.
    final_stretch: multiply the durations of the last word (tokens after the
      last space) to restore phrase-final lengthening.
    """
    dur = dur.clone()
    for b in range(dur.size(0)):
        sym = [id_to_symbol.get(int(i), "?") for i in phoneme_ids[b]]
        n = len(sym)
        if max_frames:
            for i, s in enumerate(sym):
                if s in PAUSE_SYMBOLS and (s != "_" or (i > 0 and sym[i - 1] in PAUSE_SYMBOLS - {"_"})
                                          or (i + 1 < n and sym[i + 1] in PAUSE_SYMBOLS - {"_"})):
                    continue
                dur[b, i] = dur[b, i].clamp(max=float(max_frames))
        if final_stretch != 1.0:
            spaces = [i for i, s in enumerate(sym) if s == " "]
            start = spaces[-1] + 1 if spaces else 0
            seg = dur[b, start:n]
            dur[b, start:n] = torch.where(seg > 0, torch.clamp_min(torch.round(seg * final_stretch), 1.0), seg)
    return dur


# IPA consonants that are voiceless in every language espeak covers.
VOICELESS = set("ptkqʔfθsʃçxχħhɸʂɕʍɬ") | {"ts", "tʃ", "tɕ", "ʈʂ"}


def fix_voiceless(prosody, phoneme_ids, id_to_symbol, max_vfrac=0.4):
    """prosody [b, 3, T] (lf0, vfrac, energy): force voiceless consonants to be
    unvoiced (no pitch), against buzzy voiced fricatives/stops when the sampled
    voicing crosses 0.5."""
    prosody = prosody.clone()
    for b in range(prosody.size(0)):
        for i, t in enumerate(phoneme_ids[b]):
            if id_to_symbol.get(int(t)) in VOICELESS:
                prosody[b, 1, i] = prosody[b, 1, i].clamp(max=max_vfrac)
                prosody[b, 0, i] = 0.0
    return prosody


# Vowels and sonorants: voiced in every language espeak covers (the length mark
# belongs to the vowel before it).
VOICED = set("aeiouyɑɐɒæəɘɚɛɜɝɞɨɪʊʌɔøœɯɤʏɵɶɑ̃mnŋɲɴlɫɹrɾɽʀʁjwʋɥʎʟː")
PAD = "_"


def fix_voicing(prosody, phoneme_ids, id_to_symbol, max_vfrac=0.4):
    """Make voicing follow the phoneme identity instead of the sampled value.

    The training targets come from a MAS alignment on Piper's token grid,
    which often puts a vowel's frames on the neighbouring pad token and marks
    the vowel token itself unvoiced (~12% of vowels in Expresso ex02). The
    predictor learns that, and when the durations come from another model
    (e.g. --pp-durations sdp) such a token becomes an unvoiced, buzzy vowel.

    - voiceless consonants (VOICELESS): vfrac <= max_vfrac, no pitch;
    - vowels/sonorants (VOICED) and pads between two of them: voiced, with
      log-F0 linearly interpolated from the voiced tokens around them;
    - everything else (voiced obstruents, stress marks, other pads): unchanged.
    prosody: [b, 3, T] = (lf0, vfrac, energy), lf0 standardised, 0 = unvoiced.
    """
    prosody = prosody.clone()
    for b in range(prosody.size(0)):
        sym = [id_to_symbol.get(int(t), "?") for t in phoneme_ids[b]]
        n = len(sym)
        lf0, vf = prosody[b, 0], prosody[b, 1]
        force_on = []
        for i, s in enumerate(sym):
            if s in VOICELESS:
                vf[i] = vf[i].clamp(max=max_vfrac)
                lf0[i] = 0.0
            elif s in VOICED or (
                s == PAD and 0 < i < n - 1 and sym[i - 1] in VOICED and sym[i + 1] in VOICED
            ):
                if vf[i] <= 0.5:
                    force_on.append(i)
        voiced = [i for i in range(n) if vf[i] > 0.5 and i not in force_on]
        if not voiced:
            continue
        xs = torch.tensor(voiced, dtype=lf0.dtype, device=lf0.device)
        ys = lf0[voiced]
        for i in force_on:
            j = int(torch.searchsorted(xs, torch.tensor(float(i), device=xs.device)))
            if j == 0:
                v = ys[0]
            elif j >= len(voiced):
                v = ys[-1]
            else:
                x0, x1 = xs[j - 1], xs[j]
                v = ys[j - 1] + (ys[j] - ys[j - 1]) * (i - x0) / (x1 - x0)
            lf0[i] = v
            vf[i] = 1.0
    return prosody


@torch.no_grad()
def text_hidden(model_g, phoneme_ids: torch.Tensor, lengths: torch.Tensor):
    """Frozen enc_p hidden states (before the prior projection)."""
    x, _, _, x_mask = model_g.enc_p(phoneme_ids, lengths)
    return x, x_mask


def n_params(m: nn.Module) -> float:
    return sum(p.numel() for p in m.parameters()) / 1e6
