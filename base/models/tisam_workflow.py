"""YAML contract shared by CODAvision's TiSAM page and standalone worker.

Workflow metadata stays outside native TrainConfig. XML supplies geometry only;
class identities and all visualization colors are owned by this YAML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationError,
    model_validator,
)
from tisam.config.run import TrainConfig

if __package__:
    from .tisam_metadata import read_colormap
else:
    from tisam_metadata import read_colormap


class Preprocessing(BaseModel):
    """Describe geometry and source-level splitting for tisam_data."""

    model_config = ConfigDict(extra="forbid")
    downsample: float = Field(default=1, ge=1, allow_inf_nan=False)
    xml_coordinate_origin: Literal[0, 1] = 0
    label_mapping: dict[str, str] = Field(default_factory=dict)
    overlap_order: list[str] = Field(default_factory=list)
    validation_fraction: float = Field(default=0.2, gt=0, lt=1)
    validation_seed: int = 42
    tile_stride: int | None = Field(default=None, gt=0)
    min_labeled_fraction: float = Field(default=0.01, ge=0, le=1)
    ignore_void_in_loss: bool = False


class WorkflowConfig(BaseModel):
    """Validate one user-owned configuration for training, testing and inference."""

    model_config = ConfigDict(extra="forbid")
    workflow_version: Literal[1]
    training: TrainConfig
    preprocessing: Preprocessing = Field(default_factory=Preprocessing)
    colormap: dict[str, list[StrictInt]]
    inference: dict[str, StrictInt] = Field(
        default_factory=lambda: {
            "patch_size": 1024,
            "effective_size": 800,
            "batch_size": 1,
        }
    )
    checkpoint: Path | None = None
    paths: dict[str, Path] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_workflow(self):
        """Reject contradictory label, geometry and split contracts before launching."""
        names = self.training.data.class_names
        read_colormap(self.colormap, names)
        if set(self.inference) != {"patch_size", "effective_size", "batch_size"}:
            raise ValueError(
                "inference accepts patch_size, effective_size and batch_size"
            )
        patch, effective = (
            self.inference["patch_size"],
            self.inference["effective_size"],
        )
        if (
            any(v <= 0 for v in self.inference.values())
            or patch < effective
            or (patch - effective) % 2
        ):
            raise ValueError(
                "Inference sizes must be positive; patch-effective must be nonnegative and even"
            )
        size = self.training.model.input_hw
        if size[0] != size[1] or size[0] != self.inference["patch_size"]:
            raise ValueError(
                "model.input_hw must be square and match inference.patch_size"
            )
        prep = self.preprocessing
        if prep.tile_stride is not None and prep.tile_stride > size[0]:
            raise ValueError(
                "tile_stride cannot exceed input tile size (would leave gaps)"
            )
        if prep.overlap_order and (
            len(prep.overlap_order) != len(names) - 1
            or set(prep.overlap_order) != set(names[1:])
        ):
            raise ValueError(
                "overlap_order must list every non-void class exactly once"
            )
        for target in prep.label_mapping.values():
            if target not in names:
                raise ValueError(f"Unknown label_mapping target: {target}")
        if set(self.paths) - set(INPUT_NAMES):
            raise ValueError(f"paths accepts only {', '.join(INPUT_NAMES)}")
        if any(
            (
                self.training.data.train,
                self.training.data.validation,
                self.training.data.test,
            )
        ):
            raise ValueError(
                "Leave native data train/validation/test empty; the four input directories supply data"
            )
        return self

    @property
    def directory(self):
        return self.training.out_dir / self.training.name

    def native_config(self):
        """Build native loaders with canonical labels, without changing the source YAML."""
        payload = self.training.model_dump(mode="json")
        names = self.training.data.class_names
        metadata = {
            "name": "codavision",
            "label_to_class_name": names,
            "supervision_ignored_raw_labels": [0]
            if self.preprocessing.ignore_void_in_loss
            else [],
            "metric_ignored_raw_labels": [0],
            "train_void_canvas_policy": "preserve",
        }
        for split in ("train", "validation"):
            payload["data"][split] = [
                {
                    "path": str(self.directory / "tiles" / split),
                    "metadata": metadata,
                }
            ]
        return TrainConfig.model_validate(payload)

    def model_profile(self, config_path, checkpoint=None):
        """Expose the same workflow colors/config to the existing TiSAM viewer."""
        payload = self.training.model_dump(mode="json")
        payload["colormap"] = self.colormap
        return {
            "schema_version": 1,
            "model_type": "TiSAM",
            "nm": self.training.name,
            "checkpoint": str(
                checkpoint
                or self.checkpoint
                or self.directory / "checkpoints/best.safetensors"
            ),
            "tisam_config": payload,
            "inference": self.inference,
            "device": self.training.device,
        }


INPUT_NAMES = (
    "training_images",
    "training_annotations",
    "testing_images",
    "testing_annotations",
)


def load_workflow(path):
    """Resolve all user-authored paths relative to the selected YAML file."""
    path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict) or payload.get("workflow_version") != 1:
        raise ValueError(
            "Select a TiSAM workflow YAML with workflow_version: 1. Native training settings belong under training; see examples/tisam-training.yml."
        )
    try:
        config = WorkflowConfig.model_validate(payload)
    except ValidationError as error:
        details = [
            f"{'.'.join(map(str, item['loc'])) or 'configuration'}: {item['msg']}"
            for item in error.errors()
        ]
        raise ValueError("Invalid workflow YAML:\n" + "\n".join(details)) from error

    def resolve(value):
        value = Path(value).expanduser()
        return (
            value.resolve() if value.is_absolute() else (path.parent / value).resolve()
        )

    config.training.out_dir = resolve(config.training.out_dir)
    if config.training.sam3_checkpoint:
        config.training.sam3_checkpoint = resolve(config.training.sam3_checkpoint)
    if config.checkpoint:
        config.checkpoint = resolve(config.checkpoint)
    config.paths = {key: resolve(value) for key, value in config.paths.items()}
    return config
