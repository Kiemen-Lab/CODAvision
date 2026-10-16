"""Direct-file worker for CODAvision's independent TiSAM workflow.

Runs input inspection, CPU preparation, native training and held-out slide
segmentation/evaluation. JSON status is consumed by the embedded TiSAM page.
"""

from __future__ import annotations

import argparse
import gc
import json
import traceback
from pathlib import Path

import numpy as np
import portalocker
import torch
import yaml
from PIL import Image
from tisam.inference import segment_wsi
from tisam.training.runner import train

if __package__:
    from .tisam_data import (
        AnnotationSource,
        ImageSource,
        check_inputs,
        emit,
        prepare,
        windows,
    )
    from .tisam_inference import restore_model
    from .tisam_metadata import blend_overlay, read_colormap
    from .tisam_workflow import INPUT_NAMES, load_workflow
else:
    from tisam_data import (
        AnnotationSource,
        ImageSource,
        check_inputs,
        emit,
        prepare,
        windows,
    )
    from tisam_inference import restore_model
    from tisam_metadata import blend_overlay, read_colormap
    from tisam_workflow import INPUT_NAMES, load_workflow


def write_json(path, payload):
    """Atomically publish inspection/run receipts, never a partial success marker."""
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def preview(config, pair, destination):
    """Show a bounded annotation crop using only YAML's palette."""
    with ImageSource(pair.image, config.preprocessing.downsample) as source:
        annotation = AnnotationSource(pair, config)
        size = min(1024, *source.shape)
        x = y = 0
        for regions in annotation.layers.values():
            if regions:
                x0, y0, x1, y1 = regions[0][1]
                x = max(0, min(source.shape[1] - size, int((x0 + x1 - size) / 2)))
                y = max(0, min(source.shape[0] - size, int((y0 + y1 - size) / 2)))
                break
        pixels = source.read(x, y, size, size)
        labels = annotation.read(x, y, size, size)
        palette = read_colormap(config.colormap, config.training.data.class_names)
        Image.fromarray(blend_overlay(pixels, labels, palette)).save(destination)


def run_identity(config, splits):
    """Prevent cached tiles/checkpoints being reused against changed inputs or policies."""
    payload = config.model_dump(mode="json")
    for key in ("device", "num_workers", "persistent_workers", "compile"):
        payload["training"].pop(key, None)
    payload.pop("checkpoint", None)
    return {
        "configuration": payload,
        "sources": {
            split: [pair.identity() for pair in pairs]
            for split, pairs in splits.items()
        },
    }


def check_resume(config, identity):
    """Reject incompatible output reuse before overwriting any generated files."""
    root = config.directory
    saved = root / "input-manifest.json"
    if saved.exists():
        if json.loads(saved.read_text()) != identity:
            raise ValueError(
                "Inputs or workflow settings changed. Choose a new training.name in YAML."
            )
    elif root.exists() and any(p.name != ".workflow.lock" for p in root.iterdir()):
        raise ValueError(
            "Output directory already contains a different run. Choose a new training.name in YAML."
        )


def evaluate_predictions(config):
    """Score continuous Test masks on non-void GT pixels, preserving true background."""
    names = config.training.data.class_names
    count = len(names)
    confusion = np.zeros((count, count), dtype=np.int64)
    per_image = []
    root = config.directory / "test"
    for prediction in sorted((root / "predictions").glob("*.tif")):
        current = np.zeros_like(confusion)
        with (
            ImageSource(prediction, mask=True) as pred,
            ImageSource(root / "labels" / prediction.name, mask=True) as truth,
        ):
            if pred.shape != truth.shape:
                raise ValueError(
                    f"Test output shape differs from annotation: {prediction.name}"
                )
            for x, y, w, h in windows(pred.shape, 1024):
                expected = truth.read(x, y, w, h).astype(np.int64)
                actual = pred.read(x, y, w, h).astype(np.int64)
                valid = expected != 0
                if np.any(actual >= count):
                    raise ValueError(
                        "Prediction has class IDs outside YAML class_names"
                    )
                current += np.bincount(
                    (expected[valid] * count + actual[valid]).ravel(),
                    minlength=count * count,
                ).reshape(count, count)
        confusion += current
        per_image.append(
            {"image": prediction.name, "evaluated_pixels": int(current.sum())}
        )
    tp = np.diag(confusion).astype(float)
    support, predicted = confusion.sum(1), confusion.sum(0)
    present = support[1:] > 0
    if not present.any():
        raise ValueError("Testing annotations contain no non-void evaluation pixels")
    precision = tp / np.maximum(predicted, 1)
    recall = tp / np.maximum(support, 1)
    f1 = 2 * tp / np.maximum(support + predicted, 1)
    return {
        "split": "Test",
        "class_names": names,
        "confusion_matrix": confusion.tolist(),
        "accuracy": float(tp[1:].sum() / support[1:].sum()),
        "precision": float(precision[1:][present].mean()),
        "recall": float(recall[1:][present].mean()),
        "f1": float(f1[1:][present].mean()),
        "per_class": [
            {
                "name": name,
                "support": int(support[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
            }
            for i, name in enumerate(names)
        ],
        "images": per_image,
        "ignored_ground_truth_ids": [0],
    }


def test_model(config):
    """Use the native best checkpoint and WSI assembler, then score held-out sources."""
    checkpoint = config.directory / "checkpoints/best.safetensors"
    model, data, _ = restore_model(
        checkpoint,
        config.training.device,
        config_payload=config.training.model_dump(mode="json"),
    )
    images = sorted((config.directory / "test/images").glob("*.tif"))
    for i, source in enumerate(images, 1):
        output = config.directory / "test/predictions" / source.name
        emit(
            "test",
            f"Testing {source.name} on {config.training.device}",
            i - 1,
            len(images),
        )
        if not output.exists():
            segment_wsi(
                model,
                source,
                output,
                **config.inference,
                mean=data.mean,
                std=data.std,
                resume=True,
            )
        emit("test", f"Test image {i}/{len(images)} complete", i, len(images))
    results = evaluate_predictions(config)
    write_json(config.directory / "test/metrics.json", results)
    return results


def main():
    """Execute an immutable UI request; output failures without claiming completion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "run"))
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    config = load_workflow(request["config"])
    paths = {
        key: str(Path(request["paths"][key]).expanduser().resolve())
        for key in INPUT_NAMES
    }
    config.paths = {key: Path(value) for key, value in paths.items()}
    if config.training.device.startswith("cuda"):
        device = torch.device(config.training.device)
        if (
            not torch.cuda.is_available()
            or (device.index or 0) >= torch.cuda.device_count()
        ):
            raise ValueError(
                f"Requested device {config.training.device} is unavailable; no CPU fallback was used"
            )
    splits, summary = check_inputs(config, paths)
    identity = run_identity(config, splits)
    check_resume(config, identity)
    if args.mode == "check":
        destination = args.request.parent / "preview.png"
        preview(config, splits["train"][0], destination)
        emit(
            "checked",
            f"Ready: {summary['train']} training, {summary['validation']} validation, {summary['test']} test sources",
            summary=summary,
            preview=str(destination),
            output=str(config.directory),
        )
        return
    config.directory.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(config.directory / ".workflow.lock", mode="a+b", timeout=0):
        check_resume(config, identity)
        write_json(config.directory / "input-manifest.json", identity)
        snapshot = config.directory / "workflow.yml"
        saved_workflow = config.model_dump(mode="json")
        saved_workflow["checkpoint"] = str(
            config.directory / "checkpoints/best.safetensors"
        )
        snapshot.write_text(yaml.safe_dump(saved_workflow, sort_keys=False))
        prepared = config.directory / "prepared.json"
        if not prepared.exists():
            summary = prepare(config, splits)
            write_json(prepared, summary)
        emit(
            "train",
            f"Training on {config.training.device}; native checkpoint resume enabled",
            output=str(config.directory),
        )
        train(config.native_config(), resume=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        results = test_model(config)
        emit(
            "completed",
            "Training, validation and independent testing completed",
            1,
            1,
            output=str(config.directory),
            metrics=results,
            config=str(snapshot),
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        emit("failed", str(error))
        traceback.print_exc()
        raise SystemExit(1) from error
