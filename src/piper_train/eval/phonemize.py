"""Text -> Piper phoneme ids using ``espeak-phonemizer`` (needs libespeak-ng).

Layout follows Piper: BOS, then each phoneme followed by a pad id, then EOS.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

_LOGGER = logging.getLogger(__name__)


class PiperPhonemizer:
    def __init__(self, config_json: Path, voice: Optional[str] = None):
        from espeak_phonemizer import Phonemizer

        config = json.load(open(config_json, encoding="utf-8"))
        # the eSpeak voice the checkpoint was trained with (e.g. "en" vs "en-us")
        self.voice = voice or config.get("espeak", {}).get("voice", "en-us")
        self.phonemizer = Phonemizer()
        id_map = config["phoneme_id_map"]
        self.id_map = {k: v[0] for k, v in id_map.items()}
        self.pad, self.bos, self.eos = (self.id_map[c] for c in ("_", "^", "$"))

    def phonemes(self, text: str) -> List[str]:
        out = self.phonemizer.phonemize(text, voice=self.voice, keep_clause_breakers=True)
        return list(out.replace("\n", " ").strip())

    def ids(self, text: str) -> List[int]:
        ids = [self.bos]
        for ph in self.phonemes(text):
            if ph not in self.id_map:
                _LOGGER.warning("Unknown phoneme %r dropped", ph)
                continue
            ids += [self.id_map[ph], self.pad]
        # Piper puts a pad after BOS and after every phoneme; EOS closes
        ids.insert(1, self.pad)
        ids.append(self.eos)
        return ids
