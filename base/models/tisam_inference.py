"""Run native TiSAM inference in CODAvision's Python environment.

The dialog executes this file directly, avoiding CODAvision's eager TensorFlow
and Qt imports. TiSAM owns model construction, normalization and WSI assembly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tisam import TiSAM
from tisam.checkpointing.weights import (
    apply_weights,
    config_from_metadata,
    read_checkpoint,
)
from tisam.config.data import DataConfig
from tisam.config.run import load_config
from tisam.data.wsi_dataset import read_segmentation_source_shape
from tisam.inference import segment_wsi


def inference_settings(metadata, config_path):
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
    if config_path:
        config = load_config(config_path)
        if saved_model is not None and config.model != saved_model:
            raise ValueError("Configuration disagrees with the checkpoint architecture")
        for label, saved, configured in (
            ("class names", names, config.data.class_names),
            ("mean", mean, config.data.mean),
            ("std", std, config.data.std),
        ):
            if saved is not None and tuple(saved) != tuple(configured):
                raise ValueError(f"Configuration disagrees with checkpoint {label}")
        return config.model, config.data, config.sam3_checkpoint
    if saved_model is None or names is None or mean is None or std is None:
        raise ValueError(
            "Checkpoint metadata is incomplete. Select its TiSAM YAML configuration."
        )
    data = DataConfig(class_names=names, mean=mean, std=std)
    if len(data.class_names) != saved_model.num_classes + 1:
        raise ValueError(
            "Checkpoint class names do not match the model output channels"
        )
    return saved_model, data, None


def segment(args):
    """Validate GUI inputs, restore a checkpoint and publish its mask and class legend."""
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    legend = output.with_suffix(output.suffix + ".json")
    if source == output or output.suffix.lower() not in (".tif", ".tiff"):
        raise ValueError("Choose a distinct .tif or .tiff output")
    if output.exists() or legend.exists():
        raise FileExistsError(
            "Output or class legend already exists; choose another output"
        )
    if (
        args.effective_size <= 0
        or args.patch_size < args.effective_size
        or (args.patch_size - args.effective_size) % 2
    ):
        raise ValueError("Patch size minus effective size must be nonnegative and even")
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    read_segmentation_source_shape(source)
    print("Reading checkpoint and segmentation settings…", flush=True)
    state, metadata = read_checkpoint(
        checkpoint
    )  # Safe loading only; no implicit pickle trust.
    model_config, data, sam3_checkpoint = inference_settings(metadata, args.config)
    if not np.isfinite((*data.mean, *data.std)).all():
        raise ValueError("Normalization must contain finite values")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in the selected Python environment")
    print("Loading TiSAM encoders and weights…", flush=True)
    model = TiSAM(model_config, sam3_checkpoint=sam3_checkpoint)
    apply_weights(model, state)
    del state
    model.to(device).eval()
    print(
        f"Segmenting {source.name} on {device}; {len(data.class_names)} classes (0 = void)…",
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
    with legend.open("x") as handle:
        json.dump(
            {
                "source": str(source),
                "checkpoint": str(checkpoint),
                "class_names": data.class_names,
                "mean": data.mean,
                "std": data.std,
                "model": model_config.model_dump(mode="json"),
                "patch_size": args.patch_size,
                "effective_size": args.effective_size,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    print(f"Saved mask: {output}\nSaved class legend: {legend}", flush=True)


def main():
    """Receive shell-free arguments from gui.components.tisam_dialog."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--effective-size", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=1)
    segment(parser.parse_args())


if __name__ == "__main__":
    main()
