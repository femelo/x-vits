# Keep CPU tests polite on a shared machine (e.g. while a GPU training run is active).
import os

import torch

torch.set_num_threads(int(os.environ.get("PIPER_TEST_THREADS", "4")))
