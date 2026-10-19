"""TiSAM YAML model metadata shared by CODAvision's viewer and worker.

A YAML file stores the workflow or a checkpoint reference and native config.
Workflow loading validates TrainConfig; only workers construct TiSAM models.
Loading metadata never modifies the selected YAML or loads model weights.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import yaml

if __package__:
    from .tisam_metadata import visualization_settings
    from .tisam_workflow import load_workflow
else:
    from tisam_metadata import visualization_settings
    from tisam_workflow import load_workflow


DEFAULT_INFERENCE = {"patch_size": 1024, "effective_size": 800, "batch_size": 1}


def read_yaml(path):
    """Read a mapping for saved model or native config loading."""
    try:
        payload = yaml.safe_load(Path(path).expanduser().read_text())
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("Configuration must be a YAML mapping")  # noqa: TRY004 - invalid config value
    return payload


def validate_inference(settings):
    """Validate the patch geometry shared by metadata and worker."""
    if not isinstance(settings, dict) or any(
        type(settings.get(key)) is not int or settings[key] < 1
        for key in DEFAULT_INFERENCE
    ):
        raise ValueError(
            "Patch size, effective size and batch size must be positive integers"
        )
    difference = settings["patch_size"] - settings["effective_size"]
    if difference < 0 or difference % 2:
        raise ValueError("Patch size minus effective size must be nonnegative and even")


def resolve_reference(value, directory):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Model metadata requires a checkpoint path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else directory / path).resolve()


@dataclass
class TiSAMModel:
    """CODAvision model identity plus a native config consumed by tisam_inference."""

    path: Path
    name: str
    checkpoint: Path
    config: dict
    inference: dict
    device: str
    downsample: float = 1

    @classmethod
    def load(cls, path):
        """Read YAML references/config without requiring weights for result viewing."""
        path = Path(path).expanduser().resolve()
        payload = read_yaml(path)
        downsample = 1
        if payload.get("workflow_version") == 1:
            workflow = load_workflow(path)
            payload = workflow.model_profile(path)
            downsample = workflow.preprocessing.downsample
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != 1
            or payload.get("model_type") != "TiSAM"
        ):
            raise ValueError(
                "Select a TiSAM YAML with schema_version: 1 and model_type: TiSAM"
            )
        name = payload.get("nm")
        if (
            not isinstance(name, str)
            or not name.strip()
            or any(char in name for char in '<>:"/\\|?*')
            or name != name.strip()
        ):
            raise ValueError(
                "Model name must be nonempty and contain no filename separators"
            )
        checkpoint = resolve_reference(payload.get("checkpoint"), path.parent)
        config = deepcopy(payload.get("tisam_config"))
        if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
            raise ValueError(  # noqa: TRY004 - invalid config value
                "Model metadata requires tisam_config.model and tisam_config.data"
            )
        visualization_settings(config)
        if config.get("sam3_checkpoint"):
            config["sam3_checkpoint"] = str(
                resolve_reference(config["sam3_checkpoint"], path.parent)
            )
        inference = payload.get("inference", DEFAULT_INFERENCE.copy())
        validate_inference(inference)
        device = payload.get("device", "cuda:0")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("Device must be a nonempty string such as cuda:0 or cpu")
        return cls(path, name, checkpoint, config, inference, device, downsample)

    def classification_dir(self, source_dir):
        """Use CODAvision's classification_<model-name>_<model-type> convention."""
        return (
            Path(source_dir).expanduser().resolve()
            / f"classification_{self.name}_TiSAM"
        )


def image_pairs(source_dir, output_dir):
    """Inventory top-level RGB images/slides and their CODAvision-style TIFF outputs."""
    source_dir, output_dir = (
        Path(source_dir).expanduser().resolve(),
        Path(output_dir).expanduser().resolve(),
    )
    if not source_dir.is_dir():
        raise ValueError("Choose an existing source image directory")
    if source_dir == output_dir:
        raise ValueError(
            "Classification output must be separate from the source directory"
        )
    sources = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower()
        in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".ndpi", ".svs")
    )
    if not sources:
        raise ValueError("Source directory contains no supported RGB images or slides")
    pairs, names = [], set()
    for source in sources:
        output = output_dir / f"{source.stem}.tif"
        key = output.name.casefold()
        if key in names:
            raise ValueError(
                f"Images share the output name {output.name}; use distinct source stems"
            )
        if output.exists() and not output.is_file():
            raise ValueError(f"Output path is not a file: {output}")
        names.add(key)
        pairs.append((source, output))
    return pairs
