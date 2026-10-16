"""NDPI/Aperio XML training preparation and shared TiSAM image readers.

Used by tisam_pipeline and its input check. Sources remain read-only. OpenSlide
reads slide regions; TIFF/Zarr also reads generated masks and inference images.
No CNN preprocessing runs.
"""

from __future__ import annotations

import json
import math
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import openslide
import tifffile
import zarr
from PIL import Image


def emit(stage, message, completed=None, total=None, **extra):
    """Send structured status to the embedded page through worker stdout."""
    print(
        "CODAVISION_EVENT "
        + json.dumps(
            {
                "stage": stage,
                "message": message,
                "completed": completed,
                "total": total,
                **extra,
            }
        ),
        flush=True,
    )


@dataclass
class Pair:
    """Pair source files without renaming or rewriting user annotations."""

    image: Path
    annotation: Path

    def identity(self):
        return [
            {"path": str(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in (self.image, self.annotation)
        ]


def inventory(images, annotations):
    """Pair top-level NDPI slides with same-stem Aperio XML for check_inputs."""
    images, annotations = Path(images).resolve(), Path(annotations).resolve()
    if not images.is_dir() or not annotations.is_dir():
        raise ValueError("Select existing image and annotation directories")
    image_files = {}
    for path in sorted(images.iterdir()):
        if (
            path.is_file()
            and path.suffix.lower() == ".ndpi"
            and not path.name.startswith(".")
        ):
            if path.stem in image_files:
                raise ValueError(f"Ambiguous image stem: {path.stem}")
            image_files[path.stem] = path
    if not image_files:
        raise ValueError(f"No NDPI images found in {images}")
    annotation_files = {}
    for path in annotations.iterdir():
        if (
            path.is_file()
            and path.suffix.lower() == ".xml"
            and not path.name.startswith(".")
        ):
            annotation_files.setdefault(path.stem, []).append(path)
    pairs = []
    for stem, image in image_files.items():
        candidates = annotation_files.get(stem, [])
        if len(candidates) != 1:
            raise ValueError(
                f"{image.name}: expected one matching Aperio XML, found {len(candidates)}"
            )
        pairs.append(Pair(image, candidates[0]))
    return pairs


class ImageSource:
    """Read downsampled RGB windows with matching level-zero annotation coordinates."""

    def __init__(self, path, downsample=1, mask=False):
        self.path, self.downsample, self.mask = Path(path), downsample, mask
        self.slide = self.store = None
        self.array = None
        suffix = self.path.suffix.lower()
        if suffix in {".ndpi", ".svs"}:
            self.slide = openslide.OpenSlide(str(path))
            self.width, self.height = self.slide.dimensions
        elif suffix in {".tif", ".tiff"}:
            self.store = tifffile.imread(path, aszarr=True, level=0)
            self.array = zarr.open(self.store, mode="r")
            if (mask and self.array.ndim != 2) or (
                not mask and (self.array.ndim != 3 or self.array.shape[2] != 3)
            ):
                self.close()
                raise ValueError(
                    f"Expected {'2D integer mask' if mask else 'HxWx3 RGB image'}: {path}"
                )
            self.height, self.width = self.array.shape[:2]
        else:
            with Image.open(path) as image:
                self.array = np.asarray(image if mask else image.convert("RGB")).copy()
            if mask and self.array.ndim != 2:
                raise ValueError(
                    f"Mask must contain one integer label per pixel: {path}"
                )
            self.height, self.width = self.array.shape[:2]
        if self.array is not None and (
            (mask and self.array.dtype.kind not in "ui")
            or (not mask and self.array.dtype != np.uint8)
        ):
            self.close()
            raise ValueError(f"Unsupported pixel dtype: {path}")
        self.shape = (
            math.ceil(self.height / downsample),
            math.ceil(self.width / downsample),
        )

    def read(self, x, y, width, height):
        """Read a real window; output coordinates use the selected downsample scale."""
        x0 = min(self.width - 1, round(x * self.downsample))
        y0 = min(self.height - 1, round(y * self.downsample))
        x1 = min(self.width, round((x + width) * self.downsample))
        y1 = min(self.height, round((y + height) * self.downsample))
        if self.slide:
            level = self.slide.get_best_level_for_downsample(self.downsample)
            factor = self.slide.level_downsamples[level]
            size = (
                max(1, math.ceil((x1 - x0) / factor)),
                max(1, math.ceil((y1 - y0) / factor)),
            )
            pixels = np.asarray(
                self.slide.read_region((x0, y0), level, size).convert("RGB")
            )
        else:
            pixels = np.asarray(self.array[y0:y1, x0:x1])
        if pixels.shape[:2] != (height, width):
            pixels = cv2.resize(
                pixels,
                (width, height),
                interpolation=cv2.INTER_NEAREST if self.mask else cv2.INTER_AREA,
            )
        return pixels

    def close(self):
        if self.slide:
            self.slide.close()
        if self.store:
            self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class AnnotationSource:
    """Rasterize Aperio XML for tisam_pipeline previews and prepared training data."""

    def __init__(self, pair, config):
        self.names = config.training.data.class_names
        self.prep = config.preprocessing
        self.layers = {}
        self.dtype = np.uint8 if len(self.names) <= 256 else np.uint16
        root = ET.parse(pair.annotation).getroot()
        for annotation in root.iter("Annotation"):
            original = annotation.get("Name", "")
            name = self.prep.label_mapping.get(original, original)
            if name not in self.names:
                raise ValueError(
                    f"{pair.annotation.name}: XML class {original!r} is absent from YAML; add an explicit label_mapping"
                )
            label = self.names.index(name)
            regions = self.layers.setdefault(label, [])
            for region in annotation.iter("Region"):
                points = np.asarray(
                    [
                        (float(v.attrib["X"]), float(v.attrib["Y"]))
                        for v in region.iter("Vertex")
                    ],
                    dtype=np.float64,
                )
                if len(points) == 0:
                    continue
                if len(points) == 2 and region.get("Type") == "1":
                    (x0, y0), (x1, y1) = points
                    points = np.asarray([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
                if len(points) < 3 or not np.isfinite(points).all():
                    raise ValueError(f"Invalid XML polygon: {pair.annotation.name}")
                points = (
                    points - self.prep.xml_coordinate_origin
                ) / self.prep.downsample
                bounds = (*points.min(axis=0), *points.max(axis=0))
                regions.append((points, bounds, region.get("NegativeROA", "0") == "1"))
        if not self.layers:
            raise ValueError(f"No annotation layers found: {pair.annotation}")

    def read(self, x, y, width, height):
        labels = np.zeros((height, width), dtype=self.dtype)
        order = self.prep.overlap_order or self.names[1:]
        for name in ["void", *order]:
            regions = self.layers.get(self.names.index(name), [])
            active = [
                (points, hole)
                for points, (x0, y0, x1, y1), hole in regions
                if x1 >= x and y1 >= y and x0 < x + width and y0 < y + height
            ]
            if not active:
                continue
            coverage = np.zeros((height, width), dtype=np.uint8)
            for holes in (False, True):
                for points, hole in active:
                    if hole == holes:
                        polygon = np.rint(points - [x, y]).astype(np.int32)
                        cv2.fillPoly(coverage, [polygon], 0 if hole else 1)
            labels[coverage != 0] = self.names.index(name)
        return labels


def windows(shape, size, stride=None):
    """Yield covering windows; boundary padding is handled by the tile writer."""
    stride = stride or size
    for y in range(0, shape[0], stride):
        for x in range(0, shape[1], stride):
            yield x, y, min(size, shape[1] - x), min(size, shape[0] - y)


def check_inputs(config, paths):
    """Check NDPI headers and Aperio XML classes before tisam_pipeline runs."""
    training = inventory(paths["training_images"], paths["training_annotations"])
    testing = inventory(paths["testing_images"], paths["testing_annotations"])
    if {p.image.resolve() for p in training} & {p.image.resolve() for p in testing}:
        raise ValueError("Training and testing must use separate source images")
    if len(training) < 2:
        raise ValueError(
            "At least two training source images are required for a source-level validation split"
        )
    shuffled = list(training)
    random.Random(config.preprocessing.validation_seed).shuffle(shuffled)
    count = max(
        1,
        min(
            len(training) - 1,
            round(len(training) * config.preprocessing.validation_fraction),
        ),
    )
    splits = {
        "train": shuffled[count:],
        "validation": shuffled[:count],
        "test": testing,
    }
    summary = {key: len(value) for key, value in splits.items()}
    index = 0
    for split, pairs in splits.items():
        for pair in pairs:
            index += 1
            emit(
                "check",
                f"Checking {pair.image.name}",
                index,
                len(training) + len(testing),
            )
            with ImageSource(pair.image, config.preprocessing.downsample):
                AnnotationSource(pair, config)
    return splits, summary


def atomic_tiff(path, data=None, **kwargs):
    """Publish tiled Zstd TIFFs only after successful writing; preserve RGB layout."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    tifffile.imwrite(temporary, data, tile=(256, 256), compression="zstd", **kwargs)
    temporary.replace(path)


def prepare(config, splits):
    """Write direct canonical RGB/label tiles, plus continuous test image/GT pairs."""
    size = config.training.model.input_hw[0]
    summary = {}
    for split, pairs in splits.items():
        total_saved = 0
        for number, pair in enumerate(pairs, 1):
            emit("prepare", f"CPU: {split} — {pair.image.name}", number - 1, len(pairs))
            with ImageSource(pair.image, config.preprocessing.downsample) as source:
                annotation = AnnotationSource(pair, config)
                if split == "test":
                    destination = (
                        config.directory
                        / "test"
                        / "images"
                        / (pair.image.stem + ".tif")
                    )
                    truth = config.directory / "test" / "labels" / destination.name
                    for path, reader, rgb in (
                        (destination, source, True),
                        (truth, annotation, False),
                    ):
                        if not path.exists():
                            atomic_tiff(
                                path,
                                (
                                    reader.read(x, y, w, h)
                                    for x, y, w, h in windows(source.shape, 256)
                                ),
                                shape=(*source.shape, 3) if rgb else source.shape,
                                dtype=np.uint8 if rgb else annotation.dtype,
                                photometric="rgb" if rgb else "minisblack",
                                bigtiff=True,
                            )
                    total_saved += 1
                else:
                    stride = config.preprocessing.tile_stride or size
                    total = math.ceil(source.shape[0] / stride) * math.ceil(
                        source.shape[1] / stride
                    )
                    for step, (x, y, w, h) in enumerate(
                        windows(source.shape, size, stride), 1
                    ):
                        labels = annotation.read(x, y, w, h)
                        if (
                            np.count_nonzero(labels) / (w * h)
                            < config.preprocessing.min_labeled_fraction
                        ):
                            continue
                        tile_name = f"{pair.image.stem}__y{y}_x{x}.tif"
                        rgb_path = (
                            config.directory
                            / "tiles"
                            / split
                            / "images"
                            / tile_name
                        )
                        label_path = (
                            config.directory
                            / "tiles"
                            / split
                            / "labels"
                            / tile_name
                        )
                        if not rgb_path.exists() or not label_path.exists():
                            pixels = source.read(x, y, w, h)
                            rgb = np.zeros((size, size, 3), dtype=np.uint8)
                            mask = np.zeros((size, size), dtype=annotation.dtype)
                            rgb[:h, :w] = pixels
                            mask[:h, :w] = labels
                            atomic_tiff(rgb_path, rgb, photometric="rgb")
                            atomic_tiff(label_path, mask, photometric="minisblack")
                        total_saved += 1
                        if step == total or step % 10 == 0:
                            emit(
                                "prepare",
                                f"CPU: {split} / {pair.image.name} — tile {step}/{total}",
                                step,
                                total,
                            )
            emit(
                "prepare",
                f"CPU: {split} — {number}/{len(pairs)} sources ready",
                number,
                len(pairs),
            )
        if not total_saved:
            raise ValueError(
                f"No usable {split} data; check annotations and min_labeled_fraction"
            )
        summary[split] = total_saved
    return summary


def inference_source(source, output_dir, downsample=1):
    """Prepare a bounded RGB TIFF for native inference and aligned viewer previews."""
    source, output_dir = Path(source), Path(output_dir)
    if downsample == 1 and source.suffix.lower() in {".tif", ".tiff", ".png"}:
        return source
    target = output_dir / "images" / (source.stem + ".tif")
    identity = {
        "source": str(source.resolve()),
        "size": source.stat().st_size,
        "mtime_ns": source.stat().st_mtime_ns,
        "downsample": downsample,
    }
    receipt = target.with_suffix(".source.json")
    if target.exists():
        if not receipt.exists() or json.loads(receipt.read_text()) != identity:
            raise ValueError(
                "Prepared inference image differs from its source/settings; choose a new model output name"
            )
        return target
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(identity))
    with ImageSource(source, downsample) as image:
        atomic_tiff(
            target,
            (image.read(x, y, w, h) for x, y, w, h in windows(image.shape, 256)),
            shape=(*image.shape, 3),
            dtype=np.uint8,
            photometric="rgb",
            bigtiff=True,
        )
    return target
