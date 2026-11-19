"""Stage 4: word-level emphasis as a per-token feature.

Text uses ``*word*`` markup (Expresso's convention).  Each Piper token is
mapped back to the word it came from, so a word flag becomes a per-token
vector the prosody predictor can condition on.

eSpeak phonemizes a sentence with context (weak forms, merged function words
such as "in the" -> "ɪnðə"), so the word -> token map is found by aligning
the per-word phonemization with the sentence phonemization (difflib on the
phoneme strings without stress marks or spaces).  This is language-agnostic.
"""

import difflib
import re
from typing import List, Sequence, Tuple

import numpy as np

_MARK = re.compile(r"\*([^*]+)\*")
_IGNORE = set("ˈˌ ")


def parse_marked(text: str) -> Tuple[str, List[bool]]:
    """'You *want* the shark?' -> ('You want the shark?', [F, T, F, F]).
    A marked span may cover several words; each of them is flagged."""
    plain, flags = [], []
    pos = 0
    for m in _MARK.finditer(text):
        before = text[pos : m.start()].split()
        plain += before
        flags += [False] * len(before)
        inside = m.group(1).split()
        plain += inside
        flags += [True] * len(inside)
        pos = m.end()
    rest = text[pos:].split()
    plain += rest
    flags += [False] * len(rest)
    # re-attach punctuation that followed a marked word ("*word*," -> "word,")
    out = " ".join(plain)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    return out, _fix_flags(out, plain, flags)


def _fix_flags(out: str, plain: Sequence[str], flags: Sequence[bool]) -> List[bool]:
    words = out.split()
    if len(words) == len(plain):
        return list(flags)
    # punctuation-only items were merged into the previous word
    merged, j = [], 0
    for w in words:
        f = False
        acc = ""
        while j < len(plain) and acc != w:
            acc = (acc + " " + plain[j]).strip() if acc else plain[j]
            acc = re.sub(r"\s+([,.;:!?])", r"\1", acc)
            f = f or flags[j]
            j += 1
        merged.append(f)
    return merged


def token_word_index(text: str, phonemizer) -> Tuple[List[int], List[int]]:
    """Piper token ids for ``text`` and, per token, the index of the word it
    belongs to (-1 for BOS/EOS, word-separating spaces and their pads)."""
    words = text.split()
    sent = phonemizer.phonemes(text)  # list of phoneme symbols, Piper order
    # per-word phonemes, concatenated, with the word index of every symbol
    ref, ref_w = [], []
    for wi, w in enumerate(words):
        for ph in phonemizer.phonemes(w):
            if ph not in _IGNORE:
                ref.append(ph)
                ref_w.append(wi)
    hyp_pos = [i for i, ph in enumerate(sent) if ph not in _IGNORE]
    hyp = [sent[i] for i in hyp_pos]
    word_of_sym = [-1] * len(sent)
    sm = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        for k in range(j2 - j1):
            if i2 > i1:
                src = i1 + min(i2 - i1 - 1, (k * (i2 - i1)) // max(1, j2 - j1))
                word_of_sym[hyp_pos[j1 + k]] = ref_w[src]
            elif i1 > 0:
                word_of_sym[hyp_pos[j1 + k]] = ref_w[i1 - 1]
    # stress marks take the word of the next symbol; spaces stay -1
    for i in range(len(sent) - 2, -1, -1):
        if sent[i] in "ˈˌ" and word_of_sym[i] < 0:
            word_of_sym[i] = word_of_sym[i + 1]
    ids = phonemizer.ids(text)
    # Piper layout: BOS, pad, (sym, pad)*, EOS -> each sym's pad shares its word
    tok_w = [-1, -1]
    for ph, wi in zip(sent, word_of_sym):
        if ph not in phonemizer.id_map:
            continue
        w = -1 if ph == " " else wi
        tok_w += [w, w]
    tok_w.append(-1)
    assert len(tok_w) == len(ids), (len(tok_w), len(ids))
    return ids, tok_w


N_EMPH_FEATS = 3


def emphasis_features(marked_text: str, phonemizer) -> Tuple[List[int], np.ndarray]:
    """``*word*`` markup -> (token ids, features [3, T] float32), non-zero only
    inside emphasised words:

      0  emphasised (0/1)
      1  relative position inside the word, 0 at its first token -> 1 at its last
      2  at or after the word's primary stress mark (0/1)

    so the predictors can shape the emphasis along the word (build-up to the
    stressed syllable, then release) instead of applying a flat offset.
    """
    plain, flags = parse_marked(marked_text)
    ids, tok_w = token_word_index(plain, phonemizer)
    inv = {v: k for k, v in phonemizer.id_map.items()}
    f = np.zeros((N_EMPH_FEATS, len(ids)), np.float32)
    tw = np.array(tok_w)
    for wi, flag in enumerate(flags):
        if not flag:
            continue
        pos = np.nonzero(tw == wi)[0]
        if len(pos) == 0:
            continue
        f[0, pos] = 1.0
        f[1, pos] = np.linspace(0.0, 1.0, len(pos)) if len(pos) > 1 else 0.5
        stress = [p for p in pos if inv.get(ids[p]) == "ˈ"]
        start = stress[0] if stress else pos[0]
        f[2, pos[pos >= start]] = 1.0
    return ids, f


def emphasis_vector(marked_text: str, phonemizer) -> Tuple[List[int], np.ndarray]:
    """``*word*`` markup -> (token ids, per-token emphasis 0/1 float32)."""
    plain, flags = parse_marked(marked_text)
    ids, tok_w = token_word_index(plain, phonemizer)
    emph = np.array([1.0 if (w >= 0 and flags[w]) else 0.0 for w in tok_w], np.float32)
    return ids, emph


def word_runs(flag: np.ndarray, pos: np.ndarray = None):
    """Relative position (0..1) of every emphasised token within its word.
    pos: precomputed position channel (emphasis_features[1]); otherwise each
    contiguous run of flagged tokens is treated as one word."""
    rel = np.full(len(flag), np.nan, np.float32)
    on = flag > 0.5
    if pos is not None:
        rel[on] = pos[on]
        return rel
    i = 0
    while i < len(flag):
        if on[i]:
            j = i
            while j + 1 < len(flag) and on[j + 1]:
                j += 1
            rel[i : j + 1] = np.linspace(0.0, 1.0, j - i + 1) if j > i else 0.5
            i = j + 1
        else:
            i += 1
    return rel


def apply_template(rel, template, scale=1.0):
    """Per-token (pitch offset in semitones, duration multiplier) from an
    emphasis template {'bins', 'pitch_st', 'dur_mult'} at relative positions
    ``rel`` (NaN = not emphasised -> 0, 1)."""
    nb = template["bins"]
    centers = (np.arange(nb) + 0.5) / nb
    p = np.interp(np.nan_to_num(rel, nan=0.0), centers, template["pitch_st"])
    d = np.interp(np.nan_to_num(rel, nan=0.0), centers, template["dur_mult"])
    off = np.isnan(rel)
    p = np.where(off, 0.0, p * scale).astype(np.float32)
    d = np.where(off, 1.0, np.power(d, scale)).astype(np.float32)
    return p, d


# ---- stress-anchored emphasis ------------------------------------------------

VOWELS = set("aeiouyɑɐɒæəɘɚɛɜɝɞɨɪʊʌɔøœɯɤʏɵɶ")


def tonic_offsets(ids, tok_w, id_to_symbol, flags=None):
    """Signed token offset of every token from the stressed vowel of its word
    (the first vowel after the primary stress mark; else after a secondary
    mark; else the word's first vowel).  NaN outside words, or outside
    flagged words when ``flags`` is given."""
    tw = np.asarray(tok_w)
    off = np.full(len(ids), np.nan, np.float32)
    for wi in sorted(set(tw[tw >= 0].tolist())):
        if flags is not None and not flags[wi]:
            continue
        pos = np.nonzero(tw == wi)[0]
        syms = [id_to_symbol.get(int(ids[p]), "?") for p in pos]
        marks = [k for k, s in enumerate(syms) if s == "ˈ"] or [k for k, s in enumerate(syms) if s == "ˌ"]
        start = marks[0] if marks else 0
        c = next((k for k in range(start, len(syms)) if syms[k] in VOWELS), None)
        if c is None:
            c = next((k for k, s in enumerate(syms) if s in VOWELS), 0)
        off[pos] = np.arange(len(pos)) - c
    return off


def _asym_bump(k, mu, s_left, s_right):
    s = np.where(k < mu, s_left, s_right)
    return np.exp(-0.5 * ((k - mu) / np.maximum(s, 1e-3)) ** 2)


def shape_pitch(k, p):
    """Emphasis pitch offset (semitones) at token offsets k from the stressed
    vowel: asymmetric bump of height A at mu, minus a release of depth R that
    sets in around k_r (logistic, width tau)."""
    return p["A"] * _asym_bump(k, p["mu"], p["s_l"], p["s_r"]) - p["R"] / (1 + np.exp(-(k - p["k_r"]) / max(p["tau"], 1e-3)))


def shape_logdur(k, p):
    """Emphasis log-duration multiplier: word-wide C plus an asymmetric bump B."""
    return p["C"] + p["B"] * _asym_bump(k, p["mu"], p["s_l"], p["s_r"])


def shape_energy(k, p):
    """Emphasis log-energy offset: word-wide C plus an asymmetric bump B (same form as duration)."""
    return shape_logdur(k, p)


def param_energy(offsets, params, scale=1.0, style=None, key="energy"):
    """Per-token log-energy offset (``key="tilt"``: vocal-effort offset) of the
    emphasised words from a fitted template; 0 where not emphasised or when the
    template has no such shape (older fits)."""
    p = params.get("styles", {}).get(style, params) if style else params
    e = p.get(key, params.get(key))
    if e is None:
        return np.zeros(len(offsets), np.float32)
    k = np.nan_to_num(offsets, nan=0.0)
    return np.where(np.isnan(offsets), 0.0, scale * shape_energy(k, e)).astype(np.float32)


# sonorants (besides vowels) that can carry lengthening without distortion
SONORANTS = set("mnŋɲɴlɫɹrɾɽʀʁjwʋɥʎʟː")


def apply_param_template(offsets, params, scale=1.0, style=None, symbols=None, obstruent_dur=0.3):
    """Per-token (pitch offset in semitones, duration multiplier) from a fitted
    stress-anchored template {'pitch': {...}, 'dur': {...}, 'styles': {...}};
    NaN offsets (not emphasised) -> (0, 1).

    symbols: per-token phoneme symbols; when given, obstruents (stops,
    fricatives) get only ``obstruent_dur`` of the log-duration change, since a
    stretched closure sounds wrong; pads take the class of the phoneme before.
    """
    p = params.get("styles", {}).get(style, params) if style else params
    k = np.nan_to_num(offsets, nan=0.0)
    off = np.isnan(offsets)
    pitch = np.where(off, 0.0, scale * shape_pitch(k, p["pitch"])).astype(np.float32)
    logd = scale * shape_logdur(k, p["dur"])
    if symbols is not None:
        w = np.ones(len(symbols), np.float32)
        prev = None
        for i, s in enumerate(symbols):
            cls = s if s != "_" else prev
            if cls is not None and cls not in VOWELS and cls not in SONORANTS and cls not in "ˈˌ":
                w[i] = obstruent_dur
            if s != "_":
                prev = s
        logd = logd * w
    dur = np.where(off, 1.0, np.exp(logd)).astype(np.float32)
    return pitch, dur


def tonic_features(marked_text: str, phonemizer):
    """``*word*`` markup -> (token ids, [3, T] float32): emphasised flag, signed
    offset from the word's stressed vowel / 12 (clipped to +-1), and a bump
    exp(-offset^2 / 8) near the stressed vowel.  Zero outside marked words.
    This is the input of the stress-anchored learned models."""
    plain, flags = parse_marked(marked_text)
    ids, tw = token_word_index(plain, phonemizer)
    inv = {v: k for k, v in phonemizer.id_map.items()}
    off = tonic_offsets(ids, tw, inv, flags)
    f = np.zeros((3, len(ids)), np.float32)
    on = ~np.isnan(off)
    f[0, on] = 1.0
    f[1, on] = np.clip(off[on], -12, 12) / 12
    f[2, on] = np.exp(-off[on] ** 2 / 8)
    return ids, f
