"""Run native TiSAM inference in CODAvision's Python environment.

The classification window executes this file directly, avoiding CODAvision's eager TensorFlow
and Qt imports. TiSAM owns model construction, normalization and WSI assembly.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from tisam import TiSAM
from tisam.checkpointing.weights import (
    apply_weights,
    config_from_metadata,
    read_checkpoint,
)
from tisam.config.data import DataConfig
from tisam.config.run import TrainConfig
from tisam.data.wsi_dataset import read_segmentation_source_shape
from tisam.inference import segment_wsi

# The GUI launches this worker as a file; library callers use its package name.
if __package__:
    from .tisam_metadata import read_colormap, save_overlay
    from .tisam_data import inference_source
    from .tisam_model import TiSAMModel, image_pairs, validate_inference
else:
    from tisam_metadata import read_colormap, save_overlay
    from tisam_data import inference_source
    from tisam_model import TiSAMModel, image_pairs, validate_inference


def inference_settings(
    metadata, config_path=None, require_colormap=False, config_payload=None
):
    """Resolve architecture and label/preprocessing contracts before loading encoders."""
    saved_model = config_from_metadata(metadata)
    legacy = metadata.get(
        "tisam_config_payload", metadata.get("tisam_checkpoint_config_payload", {})
    )
    if isinstance(legacy, str):
        legacy = json.loads(legacy)

    def value(key, legacy_key):
        result = metadata.get(key, legacy.get(legacy_key))
        return json.loads(result) if isinstance(result, str) else result

    names = value("class_names", "dataset_class_names")
    mean = value("mean", "dataset_mean")
    std = value("std", "dataset_std")
    if config_path or config_payload is not None:
        config_path = Path(config_path).resolve() if config_path else None
        payload = (
            deepcopy(config_payload)
            if config_payload is not None
            else yaml.safe_load(config_path.read_text())
        )
        if not isinstance(payload, dict):
            raise ValueError("TiSAM configuration must be a YAML mapping")
        colormap = payload.pop("colormap", None)
        config = TrainConfig.model_validate(payload)
        if (
            config.sam3_checkpoint is not None
            and not config.sam3_checkpoint.is_absolute()
        ):
            config.sam3_checkpoint = (
                (config_path.parent if config_path else Path.cwd())
                / config.sam3_checkpoint
            ).resolve()
        palette = (
            read_colormap(colormap, config.data.class_names)
            if require_colormap
            else None
        )
        if saved_model is not None and config.model != saved_model:
            raise ValueError("Configuration disagrees with the checkpoint architecture")
        for label, saved, configured in (
            ("class names", names, config.data.class_names),
            ("mean", mean, config.data.mean),
            ("std", std, config.data.std),
        ):
            if saved is not None and tuple(saved) != tuple(configured):
                raise ValueError(f"Configuration disagrees with checkpoint {label}")
        return config.model, config.data, config.sam3_checkpoint, palette
    if require_colormap:
        raise ValueError(
            "Save overlay requires a YAML configuration containing a colormap"
        )
    if saved_model is None or names is None or mean is None or std is None:
        raise ValueError(
            "Checkpoint metadata is incomplete. Select its TiSAM YAML configuration."
        )
    data = DataConfig(class_names=names, mean=mean, std=std)
    if len(data.class_names) != saved_model.num_classes + 1:
        raise ValueError(
            "Checkpoint class names do not match the model output channels"
        )
    return saved_model, data, None, None


def restore_model(
    checkpoint, device, config_path=None, *, require_colormap=False, config_payload=None
):
    """Load a checkpoint once for a single image or entire folder; convert config in memory."""
    print("Reading checkpoint and segmentation settings…", flush=True)
    state, metadata = read_checkpoint(
        checkpoint
    )  # Safe loading only; no implicit pickle trust.
    model_config, data, sam3_checkpoint, palette = inference_settings(
        metadata,
        config_path,
        require_colormap=require_colormap,
        config_payload=config_payload,
    )
    if not np.isfinite((*data.mean, *data.std)).all():
        raise ValueError("Normalization must contain finite values")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in the selected Python environment")
    print("Loading TiSAM encoders and weights…", flush=True)
    model = TiSAM(model_config, sam3_checkpoint=sam3_checkpoint)
    apply_weights(model, state)
    del state
    model.to(device).eval()
    return model, data, palette


def segment(args):
    """Validate GUI inputs, restore a checkpoint and publish its mask and optional overlay."""
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    overlay = Path(args.overlay).resolve() if args.overlay else None
    if source == output or output.suffix.lower() not in (".tif", ".tiff"):
        raise ValueError("Choose a distinct .tif or .tiff output")
    if output.exists():
        raise FileExistsError("Output mask already exists; choose another output")
    if overlay is not None:
        if overlay.suffix.lower() != ".jpg" or overlay in (source, output):
            raise ValueError("Choose a distinct .jpg overlay output")
        if overlay.exists():
            raise FileExistsError("Overlay already exists; choose another output")
    if (
        args.effective_size <= 0
        or args.patch_size < args.effective_size
        or (args.patch_size - args.effective_size) % 2
    ):
        raise ValueError("Patch size minus effective size must be nonnegative and even")
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    shape = read_segmentation_source_shape(source)
    if overlay is not None and max(shape[:2]) > 65500:
        raise ValueError(
            "JPG supports dimensions up to 65500 pixels; use a smaller input or disable overlay"
        )
    model, data, palette = restore_model(
        checkpoint,
        args.device or "cuda:0",
        args.config,
        require_colormap=overlay is not None,
    )
    print(
        f"Segmenting {source.name} on {args.device or 'cuda:0'}; {len(data.class_names)} classes (0 = void)…",
        flush=True,
    )
    segment_wsi(
        model,
        source,
        output,
        patch_size=args.patch_size,
        effective_size=args.effective_size,
        batch_size=args.batch_size,
        num_workers=0,
        mean=data.mean,
        std=data.std,
        resume=True,
    )
    print(f"Saved mask: {output}", flush=True)
    if overlay is not None:
        print("Saving JPG overlay…", flush=True)
        save_overlay(source, output, overlay, palette)
        print(f"Saved overlay: {overlay}", flush=True)


def segment_folder(args):
    """Use CODAvision's folder/output convention with TiSAM's native resumable pipeline."""
    profile = TiSAMModel.load(args.profile)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else profile.classification_dir(args.source_dir)
    )
    pairs = image_pairs(args.source_dir, output_dir)
    settings = {
        key: getattr(args, key) if getattr(args, key) is not None else value
        for key, value in profile.inference.items()
        if key in ("patch_size", "effective_size", "batch_size")
    }
    validate_inference(settings)
    pending = [(source, output) for source, output in pairs if not output.exists()]
    # Validate all pending sources before allocating the model.
    prepared = {}
    for source, _ in pending:
        prepared[source] = inference_source(source, output_dir, profile.downsample)
        read_segmentation_source_shape(prepared[source])
    if pending:
        if not profile.checkpoint.is_file():
            raise ValueError(f"Checkpoint not found: {profile.checkpoint}")
        model, data, _ = restore_model(
            profile.checkpoint,
            args.device or profile.device,
            config_payload=profile.config,
        )
    for index, (source, output) in enumerate(pairs, 1):
        print(f"Image {index}/{len(pairs)}: {source.name}", flush=True)
        if output.exists():
            print(f"Already classified; skipping: {output.name}", flush=True)
            continue
        segment_wsi(
            model,
            prepared[source],
            output,
            **settings,
            num_workers=0,
            mean=data.mean,
            std=data.std,
            resume=True,
        )
        print(f"Saved mask: {output}", flush=True)
    print(f"Complete: {len(pairs)} images in {output_dir}", flush=True)


def main():
    """Receive shell-free arguments from gui.components.tisam_classification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--source")
    parser.add_argument("--output")
    parser.add_argument("--profile", help="TiSAM model YAML")
    parser.add_argument("--source-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--config")
    parser.add_argument("--overlay", help="Optional JPG overlay output path")
    parser.add_argument(
        "--device", help="Override the model YAML device (default cuda:0)"
    )
    parser.add_argument("--patch-size", type=int)
    parser.add_argument("--effective-size", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    if args.profile:
        if not args.source_dir or any(
            (args.checkpoint, args.config, args.source, args.output, args.overlay)
        ):
            parser.error(
                "--profile requires --source-dir and cannot be combined with single-image options"
            )
        segment_folder(args)
    else:
        if (
            not all((args.checkpoint, args.source, args.output))
            or args.source_dir
            or args.output_dir
        ):
            parser.error("Use --profile/--source-dir or --checkpoint/--source/--output")
        for key, value in {
            "patch_size": 1024,
            "effective_size": 800,
            "batch_size": 1,
        }.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
        segment(args)


if __name__ == "__main__":
    main()
