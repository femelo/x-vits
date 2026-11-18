"""Offline tests for the eval harness (no model downloads, no GPU)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from piper_train.eval import fetch_voice, score_utmos, score_wer


class TestWer(unittest.TestCase):
    def test_perfect_and_normalised(self):
        refs = {"a.wav": "Three deathly screams froze the thieves."}
        hyps = {"a.wav": "three deathly screams froze the thieves"}
        s, per = score_wer.compute_errors(refs, hyps)
        self.assertEqual(s["wer"], 0.0)
        self.assertEqual(per["a.wav"], 0.0)

    def test_errors_counted(self):
        refs = {"a.wav": "the cat sat down"}
        hyps = {"a.wav": "the cat sat town"}
        s, _ = score_wer.compute_errors(refs, hyps)
        self.assertAlmostEqual(s["wer"], 0.25)
        self.assertGreater(s["cer"], 0.0)


class TestUtmosSummary(unittest.TestCase):
    def test_summary(self):
        s = score_utmos.summarize({"a": 3.0, "b": 4.0})
        self.assertEqual(s["n"], 2)
        self.assertAlmostEqual(s["utmos_mean"], 3.5)
        self.assertAlmostEqual(s["utmos_std"], 0.5)


class TestEvalSet(unittest.TestCase):
    def test_build_is_deterministic(self):
        import gzip

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            gz = d / "ds.jsonl.gz"
            with gzip.open(gz, "wt") as f:
                for i in range(50):
                    f.write(json.dumps({"text": f"sentence number {i} is here ok", "phoneme_ids": [1, i]}) + "\n")
            fetch_voice.build_eval_set(gz, d / "a.jsonl", 10, 1)
            fetch_voice.build_eval_set(gz, d / "b.jsonl", 10, 1)
            self.assertEqual((d / "a.jsonl").read_text(), (d / "b.jsonl").read_text())
            self.assertEqual(len((d / "a.jsonl").read_text().splitlines()), 10)


class TestPhonemize(unittest.TestCase):
    CONFIG = Path(__file__).resolve().parents[4] / "data/lessac-medium/config.json"

    def test_ids_layout_and_known_sentence(self):
        if not self.CONFIG.exists():
            self.skipTest("run piper_train.eval.fetch_voice first")
        try:
            from piper_train.eval.phonemize import PiperPhonemizer

            ph = PiperPhonemizer(self.CONFIG)
            ids = ph.ids("Three deathly screams froze the thieves.")
        except (ImportError, OSError) as e:
            self.skipTest(f"libespeak-ng unavailable: {e}")
        self.assertEqual(ids[:6], [1, 0, 126, 0, 88, 0])  # BOS pad θ pad ɹ pad
        self.assertEqual(ids[-2:], [0, 2])  # pad EOS


if __name__ == "__main__":
    unittest.main()
