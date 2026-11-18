"""End-to-end CPU test: tiny synthetic dataset -> VitsModel -> Trainer.fit.

Exercises the PL 2 manual-optimization port and the CFM loss wiring. Always CPU.
"""

import json
import tempfile
import unittest
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")

from lightning import Trainer

from piper_train.vits.lightning_model import VitsModel


def write_dataset(root: Path, n=6, n_symbols=40):
    lines = []
    g = torch.Generator().manual_seed(0)
    for i in range(n):
        frames = 48 + 4 * i
        spec = torch.rand(513, frames, generator=g)
        audio = torch.randn(1, frames * 256, generator=g) * 0.1
        torch.save(spec, root / f"{i}.spec.pt")
        torch.save(audio, root / f"{i}.audio.pt")
        ids = torch.randint(1, n_symbols, (10 + i,), generator=g).tolist()
        lines.append(
            json.dumps(
                {
                    "phoneme_ids": ids,
                    "audio_norm_path": str(root / f"{i}.audio.pt"),
                    "audio_spec_path": str(root / f"{i}.spec.pt"),
                    "text": f"utt {i}",
                }
            )
        )
    path = root / "dataset.jsonl"
    path.write_text("\n".join(lines))
    return path


SMALL = dict(
    hidden_channels=32,
    inter_channels=32,
    filter_channels=64,
    n_layers=1,
    upsample_initial_channel=32,
    segment_size=2048,
    batch_size=2,
    num_workers=0,
    validation_split=0.2,
    num_test_examples=1,
)


class TestTraining(unittest.TestCase):
    def _fit(self, **extra):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ds = write_dataset(root)
            model = VitsModel(num_symbols=40, num_speakers=1, dataset=[ds], **SMALL, **extra)
            trainer = Trainer(
                fast_dev_run=2,
                accelerator="cpu",
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )
            trainer.fit(model)
            return model

    def test_vits_two_steps(self):
        m = self._fit()
        self.assertEqual(m.model_g.use_cfm, False)

    def test_cfm_two_steps_with_flow_warmup(self):
        m = self._fit(use_cfm=True, cfm_hidden_channels=32, cfm_n_layers=2, flow_mas_warmup_steps=1)
        self.assertTrue(m.model_g.use_cfm)

    def test_cfm_updates_only_trainable_parts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ds = write_dataset(root)
            model = VitsModel(
                num_symbols=40, num_speakers=1, dataset=[ds], use_cfm=True,
                cfm_hidden_channels=32, cfm_n_layers=2, **SMALL,
            )
            flow_before = {k: v.clone() for k, v in model.model_g.flow.state_dict().items()}
            est_before = model.model_g.cfm.estimator.proj.weight.clone()
            Trainer(
                fast_dev_run=3, accelerator="cpu", logger=False,
                enable_checkpointing=False, enable_progress_bar=False,
            ).fit(model)
            for k, v in model.model_g.flow.state_dict().items():
                self.assertTrue(torch.equal(v, flow_before[k]), f"flow param {k} changed")
            self.assertFalse(torch.equal(model.model_g.cfm.estimator.proj.weight, est_before))


if __name__ == "__main__":
    unittest.main()
