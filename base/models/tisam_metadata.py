"""Shared TiSAM visualization helpers for the inference worker and result viewer.

Config colors, blending and full-resolution exports share one implementation.
This module never imports TiSAM or constructs encoders.
"""

from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


def read_colormap(colormap, class_names):
    """Validate config RGB values and order them by the model's canonical class IDs."""
    if not isinstance(colormap, dict) or not colormap:
        raise ValueError("Overlay requires a nonempty 'colormap' in the YAML config")
    missing = [name for name in class_names if name not in colormap]
    if missing:
        raise ValueError(f"Config colormap is missing classes: {', '.join(missing)}")
    colors = []
    for name in class_names:
        color = colormap[name]
        if (
            not isinstance(color, (list, tuple))
            or len(color) != 3
            or any(
                type(channel) is not int or not 0 <= channel <= 255 for channel in color
            )
        ):
            raise ValueError(
                f"Config colormap[{name!r}] must contain three RGB integers in 0..255"
            )
        colors.append(color)
    return np.asarray(colors, dtype=np.float32)


def blend_overlay(image, labels, palette):
    """Blend canonical class colors at 50%; void keeps its original RGB pixels."""
    blended = ((image.astype(np.float32) + palette[labels]) * 0.5).astype(np.uint8)
    return np.where((labels != 0)[..., None], blended, image)


def save_overlay(source, mask_path, output, palette):
    """Export native class IDs over RGB pixels as a full-resolution JPG.

    Called by the inference CLI and the viewer's export thread. The palette
    comes from config metadata; interactive preview color edits are not exported.
    """
    source, mask_path, output = Path(source), Path(mask_path), Path(output)
    if palette is None:
        raise ValueError("Overlay requires a colormap in the selected YAML config")
    if output.suffix.lower() != ".jpg":
        raise ValueError("Choose a .jpg overlay output")
    if output.exists():
        raise FileExistsError("Output already exists; choose a new filename")
    with tifffile.TiffFile(mask_path) as handle:
        shape = handle.series[0].shape
    if len(shape) != 2 or max(shape) > 65500:
        raise ValueError(
            "JPG requires a 2D mask with dimensions no larger than 65500 pixels"
        )
    if source.suffix.lower() in (".tif", ".tiff"):
        image = tifffile.imread(source)
    else:
        with Image.open(source) as handle:
            image = np.asarray(handle.convert("RGB"))
    mask = tifffile.imread(mask_path)
    if image.shape != (*mask.shape, 3) or image.dtype != np.uint8:
        raise ValueError(
            "Overlay requires matching uint8 RGB source and mask dimensions"
        )
    if (
        mask.dtype.kind not in "ui"
        or int(mask.min()) < 0
        or int(mask.max()) >= len(palette)
    ):
        raise ValueError("Mask class IDs exceed the configured colormap")
    overlay = image.copy()
    # Work in strips so blending does not allocate full-slide float arrays.
    for start in range(0, mask.shape[0], 256):
        labels = mask[start : start + 256]
        original = image[start : start + 256]
        overlay[start : start + 256] = blend_overlay(original, labels, palette)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        try:
            Image.fromarray(overlay).save(
                handle, format="JPEG", quality=95, subsampling=0
            )
        except Exception:
            handle.close()
            output.unlink()
            raise


def visualization_settings(payload):
    """Validate native config class order/colors for saved models and the viewer."""
    palette = None
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        raise ValueError("Viewer configuration requires data.class_names")  # noqa: TRY004 - invalid config value
    names = payload["data"].get("class_names")
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) for name in names)
        or names[0] != "void"
        or len(set(names)) != len(names)
    ):
        raise ValueError(
            "data.class_names must contain unique names starting with void"
        )
    if "colormap" in payload:
        palette = read_colormap(payload["colormap"], names).astype(np.uint8)
    return names, palette
