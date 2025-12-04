"""
Unit tests for tile generation configuration system.

Tests the TileGenerationConfig dataclass and configuration presets
to ensure proper behavior in different tile generation modes.
"""

import pytest
import numpy as np
from base.config import (
    TileGenerationConfig,
    MODERN_CONFIG,
    LEGACY_CONFIG,
    get_default_tile_config,
    ModelDefaults
)


class TestTileGenerationConfig:
    """Test the TileGenerationConfig dataclass."""

    def test_default_config_is_modern(self):
        """Test that default config uses modern settings."""
        config = TileGenerationConfig()
        assert config.mode == "modern"
        assert config.reduction_factor == 10
        assert config.use_disk_filter is False
        assert config.crop_rotations is False
        assert config.class_rotation_frequency == 5
        assert config.deterministic_seed == 3
        assert config.big_tile_size == 10240
        assert config.file_format == "png"

    def test_modern_config_preset(self):
        """Test MODERN_CONFIG preset has correct values."""
        assert MODERN_CONFIG.mode == "modern"
        assert MODERN_CONFIG.reduction_factor == 10
        assert MODERN_CONFIG.use_disk_filter is False
        assert MODERN_CONFIG.crop_rotations is False
        assert MODERN_CONFIG.class_rotation_frequency == 5
        assert MODERN_CONFIG.deterministic_seed == 3
        assert MODERN_CONFIG.big_tile_size == 10240
        assert MODERN_CONFIG.file_format == "png"

    def test_legacy_config_preset(self):
        """Test LEGACY_CONFIG preset has correct values."""
        assert LEGACY_CONFIG.mode == "legacy"
        assert LEGACY_CONFIG.reduction_factor == 5
        assert LEGACY_CONFIG.use_disk_filter is True
        assert LEGACY_CONFIG.crop_rotations is True
        assert LEGACY_CONFIG.class_rotation_frequency == 3
        assert LEGACY_CONFIG.deterministic_seed is None
        assert LEGACY_CONFIG.big_tile_size == 10000
        assert LEGACY_CONFIG.file_format == "tif"

    def test_custom_config_creation(self):
        """Test creating a custom configuration."""
        custom_config = TileGenerationConfig(
            mode="custom",
            reduction_factor=7,
            use_disk_filter=False,
            crop_rotations=True,
            class_rotation_frequency=4,
            deterministic_seed=42,
            big_tile_size=12000,
            file_format="jpg"
        )

        assert custom_config.mode == "custom"
        assert custom_config.reduction_factor == 7
        assert custom_config.use_disk_filter is False
        assert custom_config.crop_rotations is True
        assert custom_config.class_rotation_frequency == 4
        assert custom_config.deterministic_seed == 42
        assert custom_config.big_tile_size == 12000
        assert custom_config.file_format == "jpg"

    def test_config_immutability_with_dataclass(self):
        """Test that configs can be modified (dataclasses are mutable by default)."""
        config = TileGenerationConfig()
        original_reduction = config.reduction_factor

        # Modify the config
        config.reduction_factor = 5

        # Verify modification worked
        assert config.reduction_factor == 5
        assert config.reduction_factor != original_reduction

    def test_preset_configs_are_independent(self):
        """Test that preset configs don't share state."""
        # Verify they have different values
        assert MODERN_CONFIG.reduction_factor != LEGACY_CONFIG.reduction_factor
        assert MODERN_CONFIG.use_disk_filter != LEGACY_CONFIG.use_disk_filter
        assert MODERN_CONFIG.crop_rotations != LEGACY_CONFIG.crop_rotations
        assert MODERN_CONFIG.class_rotation_frequency != LEGACY_CONFIG.class_rotation_frequency
        assert MODERN_CONFIG.deterministic_seed != LEGACY_CONFIG.deterministic_seed
        assert MODERN_CONFIG.big_tile_size != LEGACY_CONFIG.big_tile_size
        assert MODERN_CONFIG.file_format != LEGACY_CONFIG.file_format


class TestConfigParameters:
    """Test individual config parameters and their effects."""

    def test_reduction_factor_values(self):
        """Test that reduction factor accepts valid values."""
        # Valid values
        config_5 = TileGenerationConfig(reduction_factor=5)
        config_10 = TileGenerationConfig(reduction_factor=10)

        assert config_5.reduction_factor == 5
        assert config_10.reduction_factor == 10

    def test_disk_filter_boolean(self):
        """Test disk filter is properly boolean."""
        config_true = TileGenerationConfig(use_disk_filter=True)
        config_false = TileGenerationConfig(use_disk_filter=False)

        assert config_true.use_disk_filter is True
        assert config_false.use_disk_filter is False

    def test_crop_rotations_boolean(self):
        """Test crop rotations is properly boolean."""
        config_true = TileGenerationConfig(crop_rotations=True)
        config_false = TileGenerationConfig(crop_rotations=False)

        assert config_true.crop_rotations is True
        assert config_false.crop_rotations is False

    def test_class_rotation_frequency_values(self):
        """Test class rotation frequency accepts valid values."""
        config_3 = TileGenerationConfig(class_rotation_frequency=3)
        config_5 = TileGenerationConfig(class_rotation_frequency=5)

        assert config_3.class_rotation_frequency == 3
        assert config_5.class_rotation_frequency == 5

    def test_deterministic_seed_optional(self):
        """Test deterministic seed can be None or int."""
        config_none = TileGenerationConfig(deterministic_seed=None)
        config_int = TileGenerationConfig(deterministic_seed=42)

        assert config_none.deterministic_seed is None
        assert config_int.deterministic_seed == 42

    def test_big_tile_size_values(self):
        """Test big tile size accepts valid values."""
        config_10000 = TileGenerationConfig(big_tile_size=10000)
        config_10240 = TileGenerationConfig(big_tile_size=10240)

        assert config_10000.big_tile_size == 10000
        assert config_10240.big_tile_size == 10240

    def test_file_format_values(self):
        """Test file format accepts valid values."""
        config_tif = TileGenerationConfig(file_format="tif")
        config_png = TileGenerationConfig(file_format="png")
        config_jpg = TileGenerationConfig(file_format="jpg")

        assert config_tif.file_format == "tif"
        assert config_png.file_format == "png"
        assert config_jpg.file_format == "jpg"


class TestConfigModeComparison:
    """Test differences between modern and legacy modes."""

    def test_reduction_factor_difference(self):
        """Test that legacy uses finer reduction factor than modern."""
        assert LEGACY_CONFIG.reduction_factor < MODERN_CONFIG.reduction_factor
        assert LEGACY_CONFIG.reduction_factor == 5
        assert MODERN_CONFIG.reduction_factor == 10

    def test_disk_filter_difference(self):
        """Test that legacy uses disk filter, modern does not."""
        assert LEGACY_CONFIG.use_disk_filter is True
        assert MODERN_CONFIG.use_disk_filter is False

    def test_crop_rotations_difference(self):
        """Test that legacy crops rotations, modern does not."""
        assert LEGACY_CONFIG.crop_rotations is True
        assert MODERN_CONFIG.crop_rotations is False

    def test_rotation_frequency_difference(self):
        """Test that legacy rotates more frequently than modern."""
        assert LEGACY_CONFIG.class_rotation_frequency < MODERN_CONFIG.class_rotation_frequency
        assert LEGACY_CONFIG.class_rotation_frequency == 3
        assert MODERN_CONFIG.class_rotation_frequency == 5

    def test_deterministic_seed_difference(self):
        """Test that legacy uses diverse runs, modern is deterministic."""
        assert LEGACY_CONFIG.deterministic_seed is None
        assert MODERN_CONFIG.deterministic_seed == 3

    def test_file_format_difference(self):
        """Test that legacy uses TIFF, modern uses PNG."""
        assert LEGACY_CONFIG.file_format == "tif"
        assert MODERN_CONFIG.file_format == "png"


class TestEnvironmentVariableSupport:
    """Test environment variable override functionality."""

    def test_default_mode_from_config(self, monkeypatch):
        """Test that default comes from ModelDefaults when no env var is set."""
        # Ensure environment variable is not set
        monkeypatch.delenv('CODAVISION_TILE_GENERATION_MODE', raising=False)

        config = get_default_tile_config()
        assert config.mode == ModelDefaults.TILE_GENERATION_MODE

    def test_environment_variable_modern(self, monkeypatch):
        """Test that environment variable can set modern mode."""
        monkeypatch.setenv('CODAVISION_TILE_GENERATION_MODE', 'modern')

        config = get_default_tile_config()
        assert config.mode == 'modern'
        assert config.reduction_factor == 10

    def test_environment_variable_legacy(self, monkeypatch):
        """Test that environment variable can set legacy mode."""
        monkeypatch.setenv('CODAVISION_TILE_GENERATION_MODE', 'legacy')

        config = get_default_tile_config()
        assert config.mode == 'legacy'
        assert config.reduction_factor == 5

    def test_environment_variable_case_insensitive(self, monkeypatch):
        """Test that environment variable is case insensitive."""
        monkeypatch.setenv('CODAVISION_TILE_GENERATION_MODE', 'LEGACY')

        config = get_default_tile_config()
        assert config.mode == 'legacy'

    def test_environment_variable_invalid_raises_error(self, monkeypatch):
        """Test that invalid values raise ValueError."""
        monkeypatch.setenv('CODAVISION_TILE_GENERATION_MODE', 'invalid')

        with pytest.raises(ValueError, match="Invalid TILE_GENERATION_MODE"):
            get_default_tile_config()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
