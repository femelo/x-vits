import argparse
import json
import logging
from pathlib import Path

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from .vits.lightning_model import VitsModel

_LOGGER = logging.getLogger(__package__)


def _exclude_splines_from_compile() -> None:
    """Keep the SDP's rational-quadratic spline transforms out of the compiled
    graph: their data-dependent control flow causes graph breaks/recompiles.
    Applied only for compiled training runs (never at import), since the
    dynamo wrapper would break torch.jit.trace-based ONNX export."""
    from .vits import modules, transforms

    for name in (
        "piecewise_rational_quadratic_transform",
        "unconstrained_rational_quadratic_spline",
        "rational_quadratic_spline",
    ):
        setattr(transforms, name, torch.compiler.disable(getattr(transforms, name)))
    # modules.py imported the function by value; rebind its copy too.
    modules.piecewise_rational_quadratic_transform = torch.compiler.disable(
        modules.piecewise_rational_quadratic_transform
    )


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir", required=True, help="Path to pre-processed dataset directory"
    )
    parser.add_argument(
        "--checkpoint-epochs",
        type=int,
        help="Save checkpoint every N epochs (default: 1)",
    )
    parser.add_argument(
        "--quality",
        default="medium",
        choices=("x-low", "medium", "high"),
        help="Quality/size of model (default: medium)",
    )
    parser.add_argument(
        "--resume_from_single_speaker_checkpoint",
        help="For multi-speaker models only. Converts a single-speaker checkpoint to multi-speaker and resumes training",
    )
    parser.add_argument(
        "--zero-init-speaker-cond",
        action="store_true",
        help="with --resume_from_single_speaker_checkpoint: start the new speaker-conditioning "
        "layers at zero, so training starts from exactly the single-speaker voice",
    )
    # Trainer options (PL >= 2 no longer provides Trainer.add_argparse_args)
    parser.add_argument("--max_epochs", type=int, default=-1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--default_root_dir")
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--resume_from_checkpoint", help="Full resume (optimizers too)")
    parser.add_argument(
        "--pretrained-checkpoint",
        help="Initialise weights from a checkpoint (non-strict; optimizers are not "
        "restored). Use this to fine-tune a VITS checkpoint with --use-cfm.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the generator and discriminators in place "
        "(checkpoint keys unchanged); the SDP spline transforms stay eager.",
    )
    parser.add_argument(
        "--save-top-k",
        type=int,
        default=1,
        help="periodic checkpoints to keep (-1 = all; default 1 = latest only)",
    )
    parser.add_argument(
        "--compile-mode",
        default="max-autotune-no-cudagraphs",
        help="torch.compile mode (default: max-autotune-no-cudagraphs)",
    )
    parser.add_argument(
        "--compile-scope",
        choices=["full", "dec"],
        default="full",
        help="full: whole model_g + model_d; dec: only decoder + discriminators",
    )
    VitsModel.add_model_specific_args(parser)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    _LOGGER.debug(args)

    args.dataset_dir = Path(args.dataset_dir)
    if not args.default_root_dir:
        args.default_root_dir = args.dataset_dir

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    config_path = args.dataset_dir / "config.json"
    dataset_path = args.dataset_dir / "dataset.jsonl"

    with open(config_path, "r", encoding="utf-8") as config_file:
        # See preprocess.py for format
        config = json.load(config_file)
        num_symbols = int(config["num_symbols"])
        num_speakers = int(config["num_speakers"])
        sample_rate = int(config["audio"]["sample_rate"])

    callbacks = []
    if args.checkpoint_epochs is not None:
        callbacks.append(
            ModelCheckpoint(every_n_epochs=args.checkpoint_epochs, save_top_k=args.save_top_k)
        )
        _LOGGER.debug(
            "Checkpoints will be saved every %s epoch(s)", args.checkpoint_epochs
        )

    devices = args.devices
    if isinstance(devices, str) and devices.isdigit():
        devices = int(devices)
    trainer = Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        accelerator=args.accelerator,
        devices=devices,
        precision=args.precision,
        default_root_dir=args.default_root_dir,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=callbacks or None,
    )

    dict_args = vars(args)
    if args.quality == "x-low":
        dict_args["hidden_channels"] = 96
        dict_args["inter_channels"] = 96
        dict_args["filter_channels"] = 384
    elif args.quality == "high":
        dict_args["resblock"] = "1"
        dict_args["resblock_kernel_sizes"] = (3, 7, 11)
        dict_args["resblock_dilation_sizes"] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        )
        dict_args["upsample_rates"] = (8, 8, 2, 2)
        dict_args["upsample_initial_channel"] = 512
        dict_args["upsample_kernel_sizes"] = (16, 16, 4, 4)

    model = VitsModel(
        num_symbols=num_symbols,
        num_speakers=num_speakers,
        sample_rate=sample_rate,
        dataset=[dataset_path],
        **dict_args,
    )

    if args.resume_from_single_speaker_checkpoint:
        assert (
            num_speakers > 1
        ), "--resume_from_single_speaker_checkpoint is only for multi-speaker models. Use --resume_from_checkpoint for single-speaker models."

        # Load single-speaker checkpoint
        _LOGGER.debug(
            "Resuming from single-speaker checkpoint: %s",
            args.resume_from_single_speaker_checkpoint,
        )
        ck_single = torch.load(args.resume_from_single_speaker_checkpoint, map_location="cpu", weights_only=False)
        sd_single = ck_single["state_dict"]
        g_dict = {k[len("model_g."):]: v for k, v in sd_single.items() if k.startswith("model_g.")}
        d_dict = {k[len("model_d."):]: v for k, v in sd_single.items() if k.startswith("model_d.")}
        for key in list(g_dict.keys()):
            # Remove keys that can't be copied over due to missing speaker embedding
            if (
                key.startswith("dec.cond")
                or key.startswith("dp.cond")
                or ("enc.cond_layer" in key)
            ):
                g_dict.pop(key, None)

        # Copy over the multi-speaker model, excluding keys related to the
        # speaker embedding (which is missing from the single-speaker model).
        load_state_dict(model.model_g, g_dict)
        load_state_dict(model.model_d, d_dict)
        if args.zero_init_speaker_cond:
            n = 0
            for name, mod in model.model_g.named_modules():
                if name.endswith("cond") or name.endswith("cond_layer"):
                    # weight-normed layers: zero the gain (zeroing the direction divides by 0)
                    names = dict(mod.named_parameters(recurse=False))
                    for pn in (["weight_g"] if "weight_g" in names else ["weight"]) + ["bias"]:
                        if pn in names:
                            torch.nn.init.zeros_(names[pn])
                            n += names[pn].numel()
            _LOGGER.info("Speaker conditioning starts at zero (%d parameters)", n)
        _LOGGER.info(
            "Successfully converted single-speaker checkpoint to multi-speaker"
        )

    if args.pretrained_checkpoint:
        ckpt = torch.load(
            args.pretrained_checkpoint, map_location="cpu", weights_only=False
        )
        result = model.load_state_dict(ckpt["state_dict"], strict=False)
        _LOGGER.info(
            "Loaded pretrained weights: %s missing, %s unexpected",
            len(result.missing_keys),
            len(result.unexpected_keys),
        )
        _LOGGER.debug("Missing keys (new modules): %s", result.missing_keys)

    if args.compile:
        torch.set_float32_matmul_precision("high")
        if args.compile_scope == "full":
            _exclude_splines_from_compile()
            model.model_g.compile(mode=args.compile_mode)
        else:
            model.model_g.dec.compile(mode=args.compile_mode)
        model.model_d.compile(mode=args.compile_mode)
        _LOGGER.info("torch.compile: scope=%s mode=%s", args.compile_scope, args.compile_mode)

    # Our checkpoints store pathlib hparams, which torch>=2.6 refuses under
    # weights_only=True; they are locally produced, so full unpickling is fine.
    trainer.fit(model, ckpt_path=args.resume_from_checkpoint, weights_only=False)


def load_state_dict(model, saved_state_dict):
    state_dict = model.state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        if k in saved_state_dict:
            # Use saved value
            new_state_dict[k] = saved_state_dict[k]
        else:
            # Use initialized value
            _LOGGER.debug("%s is not in the checkpoint", k)
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
