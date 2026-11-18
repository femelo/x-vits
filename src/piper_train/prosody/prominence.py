"""Stage 4b: word prominence from the recordings, and a small student that
predicts it from text.

Labels (no language model, no manual marks): for every word of every training
recording, an acoustic prominence score from the stage 1 targets:

  pitch  - highest voiced log-F0 in the word, semitones above the sentence median
  length - mean log duration of the word's vowels relative to the corpus mean
           of the same vowel symbol (so long vowels are not "prominent" per se)

Both are z-scored within the sentence and summed; a word is prominent when its
score is >= ``z_thr``.  This works for any voice/language with stage 1 targets.

Student: a head on the voice's frozen text encoder (``enc_p`` hidden states,
the same input the stage 3 predictor uses) -> per-token logit, pooled per word
(max).  ~1M parameters, phoneme based, language-agnostic.
"""

from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn

from .emphasis import VOWELS


def vowel_duration_means(utts: Dict[str, dict], rows, id_to_symbol) -> Dict[str, float]:
    """Corpus mean log-duration per vowel symbol (frames + 0.5)."""
    acc: Dict[str, List[float]] = {}
    for utt, ids in rows:
        d = utts[utt]["dur"]
        for t, x in zip(ids, d):
            s = id_to_symbol.get(int(t))
            if s in VOWELS:
                acc.setdefault(s, []).append(float(np.log(x + 0.5)))
    return {s: float(np.mean(v)) for s, v in acc.items()}


def word_prominence(utt: dict, ids: Sequence[int], tok_w: Sequence[int], id_to_symbol, vmeans) -> np.ndarray:
    """Per-word prominence score (sum of within-sentence z-scores of pitch peak
    and vowel lengthening); NaN for words without voiced/vowel tokens."""
    tw = np.asarray(tok_w)
    n_words = int(tw.max()) + 1 if (tw >= 0).any() else 0
    st = 12 * np.nan_to_num(utt["lf0"]) / np.log(2)
    voiced = utt["vfrac"] > 0.5
    valid = tw >= 0
    med = np.median(st[voiced & valid]) if (voiced & valid).any() else 0.0
    pitch = np.full(n_words, np.nan)
    length = np.full(n_words, np.nan)
    for w in range(n_words):
        m = tw == w
        if (m & voiced).any():
            pitch[w] = st[m & voiced].max() - med
        vl = [np.log(utt["dur"][i] + 0.5) - vmeans[id_to_symbol[int(ids[i])]]
              for i in np.nonzero(m)[0] if id_to_symbol.get(int(ids[i])) in vmeans]
        # vowel time often sits on the pad after the vowel token: include it
        vl += [np.log(utt["dur"][i + 1] + 0.5) - vmeans[id_to_symbol[int(ids[i])]]
               for i in np.nonzero(m)[0] if id_to_symbol.get(int(ids[i])) in vmeans and i + 1 < len(ids)]
        if vl:
            length[w] = float(np.mean(vl))

    def z(a):
        ok = ~np.isnan(a)
        if ok.sum() < 2:
            return np.where(ok, 0.0, np.nan)
        return (a - np.nanmean(a)) / (np.nanstd(a) + 1e-6)

    return np.nansum(np.stack([z(pitch), z(length)]), 0) * np.where(np.isnan(pitch) & np.isnan(length), np.nan, 1.0)


class ProminenceHead(nn.Module):
    """enc_p hidden [b, C, T] -> per-token logits [b, T]."""

    def __init__(self, text_channels=192, hidden=192, n_conv=3, n_attn=2, heads=2, p_dropout=0.1):
        super().__init__()
        self.inp = nn.Conv1d(text_channels, hidden, 1)
        self.convs = nn.ModuleList(nn.Conv1d(hidden, hidden, 5, padding=2) for _ in range(n_conv))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(n_conv))
        self.attn = nn.ModuleList(
            nn.TransformerEncoderLayer(hidden, heads, 2 * hidden, p_dropout, batch_first=True, norm_first=True)
            for _ in range(n_attn)
        )
        self.drop = nn.Dropout(p_dropout)
        self.out = nn.Linear(hidden, 1)
        self.kw = dict(text_channels=text_channels, hidden=hidden, n_conv=n_conv, n_attn=n_attn, heads=heads, p_dropout=p_dropout)

    def forward(self, x, mask):
        h = self.inp(x) * mask
        for conv, norm in zip(self.convs, self.norms):
            h = h + self.drop(torch.relu(conv(h))) * mask
            h = norm(h.transpose(1, 2)).transpose(1, 2)
        h = h.transpose(1, 2)
        pad = mask.squeeze(1) < 0.5
        for layer in self.attn:
            h = layer(h, src_key_padding_mask=pad)
        return self.out(h).squeeze(-1)  # [b, T]


def pool_words(token_logits: torch.Tensor, tok_w: torch.Tensor, n_words: int) -> torch.Tensor:
    """[T] token logits, [T] word index (-1 = none) -> [n_words] word logits (max)."""
    out = token_logits.new_full((n_words,), -1e4)
    m = tok_w >= 0
    return out.scatter_reduce(0, tok_w[m], token_logits[m], reduce="amax", include_self=True)
