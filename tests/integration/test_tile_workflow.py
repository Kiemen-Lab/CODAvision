"""
Integration tests for tile generation workflow.

Tests comparing modern vs legacy tile generation modes with test-scaled
configurations (see TILE_GENERATION_ANALYSIS.md for details).
"""

import pytest
import numpy as np
import os
import tempfile
import shutil
import pickle
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple

from base.data.tiles import (
    create_training_tiles,
    create_training_tiles_modern,
    create_training_tiles_legacy
)
from base.config import (
    TileGenerationConfig,
    MODERN_CONFIG,
    LEGACY_CONFIG,
    get_default_tile_config
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_image_500x500() -> np.ndarray:
    """
    Create a synthetic 512x512 H&E-like image for testing.

    Creates an image with tissue-like regions (pinkish/purplish) and
    background regions (white/near-white).

    Note: Size increased to 512x512 to ensure legacy mode's disk filter (51x51)
    and padding (20px) have sufficient space to operate without pathological edge cases.

    Returns:
        512x512x3 RGB image as numpy array
    """
    image = np.ones((512, 512, 3), dtype=np.uint8) * 240  # Near-white background

    # Add tissue-like regions with H&E-like colors (scaled to 512x512)
    # Region 1: Pink/eosin-like region (top-left quadrant)
    image[40:200, 40:200] = [220, 150, 180]  # Pinkish

    # Region 2: Purple/hematoxylin-like region (top-right quadrant)
    image[40:200, 300:460] = [180, 150, 200]  # Purplish

    # Region 3: Mixed region (bottom-center)
    image[300:460, 140:360] = [200, 160, 190]  # Mixed pink-purple

    # Add some noise to make it more realistic
    noise = np.random.randint(-10, 10, size=(512, 512, 3), dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return image


@pytest.fixture
def test_annotations_3_classes() -> np.ndarray:
    """
    Create synthetic 3-class annotation pixel counts matching the test image.

    Creates annotation counts with three distinct classes:
    - Class 1: Background/Normal tissue
    - Class 2: Region of interest 1 (e.g., cellular region)
    - Class 3: Region of interest 2 (e.g., stroma)

    Note: Updated to match 512x512 image size.

    Returns:
        2D array with shape (1, 3) containing pixel counts for each class:
        [[class1_pixels, class2_pixels, class3_pixels]]
    """
    # Create 2D annotation mask to calculate pixel counts (512x512 to match image)
    mask = np.ones((512, 512), dtype=np.uint8)  # Class 1 (background)

    # Class 2 regions (cellular regions) - scaled to 512x512
    mask[40:200, 40:200] = 2
    mask[40:200, 300:460] = 2

    # Class 3 regions (stroma-like regions) - scaled to 512x512
    mask[300:460, 140:360] = 3

    # Count pixels for each class
    class1_pixels = np.sum(mask == 1)
    class2_pixels = np.sum(mask == 2)
    class3_pixels = np.sum(mask == 3)

    # Return as 2D array with shape (1, 3) - one image, three classes
    return np.array([[class1_pixels, class2_pixels, class3_pixels]], dtype=np.uint32)


@pytest.fixture
def temp_model_directory():
    """
    Create a temporary directory for model outputs.

    Yields:
        Path to temporary directory

    Cleanup:
        Removes directory and all contents after test
    """
    temp_dir = tempfile.mkdtemp(prefix="codavision_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_image_list(test_image_500x500, temp_model_directory) -> Dict[str, List[str]]:
    """
    Create image_list structure with saved test image and annotation mask.

    Creates the expected directory structure:
        temp_dir/
            im/          <- Images directory
                test_image.png
            label/       <- Annotation masks directory
                test_image.png

    Args:
        test_image_500x500: Test image fixture
        temp_model_directory: Temporary directory fixture

    Returns:
        Dictionary with 'tile_name' and 'tile_pth' lists
    """
    # Create both im/ and label/ directories (expected by tile generation)
    image_dir = os.path.join(temp_model_directory, "im")
    label_dir = os.path.join(temp_model_directory, "label")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    image_filename = "test_image.png"

    # Save test image to im/ directory
    image_path = os.path.join(image_dir, image_filename)
    Image.fromarray(test_image_500x500).save(image_path)

    # Create and save annotation mask to label/ directory
    # Match the structure from test_annotations_3_classes fixture (512x512)
    mask = np.ones((512, 512), dtype=np.uint8)  # Class 1 (background)

    # Class 2 regions (cellular regions) - scaled to 512x512
    mask[40:200, 40:200] = 2
    mask[40:200, 300:460] = 2

    # Class 3 regions (stroma-like regions) - scaled to 512x512
    mask[300:460, 140:360] = 3

    label_path = os.path.join(label_dir, image_filename)
    Image.fromarray(mask).save(label_path)

    # Create image_list structure with proper format:
    # tile_pth points to the im/ directory (where images are stored)
    # The validation code expects tile_pth to be parent/im/ so it can
    # derive parent_dir = os.path.dirname(tile_pth) to find parent/label/
    image_list = {
        "tile_name": [image_filename],
        "tile_pth": [image_dir]  # Points to im/ directory
    }

    return image_list


@pytest.fixture
def test_legacy_config() -> TileGenerationConfig:
    """
    Create a test-specific legacy configuration scaled for small test images.

    This configuration adapts LEGACY_CONFIG parameters for 512x512 test images:
    - reduction_factor: 5 → 2 (allows finer detail with small images)
    - use_disk_filter: True → False (51x51 filter too large for 512x512)
    - big_tile_size: 10000 → 1024 (scaled for test image size)
    - deterministic_seed: None → 42 (ensures reproducible tests)

    The standard LEGACY_CONFIG is designed for production images (10000x10000+)
    and is mathematically incompatible with small test images. This scaled
    configuration preserves the legacy mode behavior while working correctly
    with test fixtures.

    Returns:
        TileGenerationConfig scaled for test images
    """
    return TileGenerationConfig(
        mode="test_legacy",
        reduction_factor=2,  # Scaled from 5 to 2 for small images
        use_disk_filter=False,  # Disabled - 51x51 filter too large
        crop_rotations=True,  # Keep legacy behavior
        class_rotation_frequency=3,  # Keep legacy behavior
        deterministic_seed=42,  # Ensure reproducible tests
        big_tile_size=1024,  # Scaled from 10000 to 1024
        file_format="tif"  # Keep legacy format
    )


@pytest.fixture
def test_modern_config() -> TileGenerationConfig:
    """
    Create a test-specific modern configuration scaled for small test images.

    This configuration adapts MODERN_CONFIG parameters for 512x512 test images:
    - big_tile_size: 10240 → 2048 (scaled for test image size and speed)
    - Other parameters kept the same as MODERN_CONFIG

    The standard MODERN_CONFIG is designed for production images (10000x10000+)
    and while it works with small images, it's unnecessarily slow for testing.
    This scaled configuration preserves the modern mode behavior while completing
    tests in ~30 seconds instead of ~180 seconds.

    Returns:
        TileGenerationConfig scaled for test images
    """
    return TileGenerationConfig(
        mode="test_modern",
        reduction_factor=10,  # Keep modern behavior (coarse placement)
        use_disk_filter=False,  # Keep modern behavior (no disk filter)
        crop_rotations=False,  # Keep modern behavior (no rotation cropping)
        class_rotation_frequency=5,  # Keep modern behavior
        deterministic_seed=3,  # Keep modern behavior (reproducible)
        big_tile_size=2048,  # Scaled from 10240 to 2048 for faster tests
        file_format="png"  # Keep modern format
    )


# ============================================================================
# Helper Functions
# ============================================================================

def create_model_metadata(model_path: str, num_classes: int = 3) -> None:
    """
    Create model metadata (net.pkl) file in the specified directory.

    Note: Updated for 512x512 test images to ensure legacy mode's disk filter
    and padding have sufficient space to operate correctly.

    Args:
        model_path: Directory to create the metadata file in
        num_classes: Number of annotation classes (default: 3)
    """
    metadata = {
        'sxy': 512,  # Tile size (matches test image size)
        'nblack': num_classes,  # Number of classes
        'classNames': [f'Class{i+1}' for i in range(num_classes)],
        'ntrain': 1,  # Minimum tiles for testing (avoids heavy oversampling)
        'nvalidate': 0  # Skip validation tiles to reduce test time
    }

    metadata_path = os.path.join(model_path, 'net.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.timeout(60)  # 1 minute timeout for integration tests
class TestTileWorkflowModernVsLegacy:
    """Test that modern and legacy modes produce different outputs."""

    def test_modern_and_legacy_produce_different_files(
        self,
        test_annotations_3_classes,
        test_image_list,
        test_modern_config,
        test_legacy_config,
        temp_model_directory
    ):
        """Test that modern and legacy modes create files with different formats.

        Note: This test uses test_modern_config and test_legacy_config (scaled
        for 512x512 test images) instead of MODERN_CONFIG and LEGACY_CONFIG
        (designed for production-scale images 10000x10000+). The test configs
        scale parameters appropriately while preserving mode behavior characteristics
        and completing in ~30 seconds instead of ~180 seconds.
        """
        # Create separate directories for modern and legacy outputs
        modern_dir = os.path.join(temp_model_directory, "modern")
        legacy_dir = os.path.join(temp_model_directory, "legacy")
        os.makedirs(modern_dir, exist_ok=True)
        os.makedirs(legacy_dir, exist_ok=True)

        # Create metadata files for both directories
        create_model_metadata(modern_dir)
        create_model_metadata(legacy_dir)

        # Run modern mode with test-specific scaled configuration
        create_training_tiles(
            model_path=modern_dir,
            annotations=test_annotations_3_classes,
            image_list=test_image_list,
            create_new_tiles=True,
            config=test_modern_config
        )

        # Run legacy mode with test-specific scaled configuration
        create_training_tiles(
            model_path=legacy_dir,
            annotations=test_annotations_3_classes,
            image_list=test_image_list,
            create_new_tiles=True,
            config=test_legacy_config
        )

        # Check that files were created in both directories
        modern_tiles = list(Path(modern_dir).rglob("HE*.*"))
        legacy_tiles = list(Path(legacy_dir).rglob("HE*.*"))

        assert len(modern_tiles) > 0, "Modern mode should create tiles"
        assert len(legacy_tiles) > 0, "Legacy mode should create tiles"

        # Check file formats (modern=png, legacy=tif)
        modern_formats = {f.suffix for f in modern_tiles}
        legacy_formats = {f.suffix for f in legacy_tiles}

        assert ".png" in modern_formats, "Modern mode should create PNG files"
        assert ".tif" in legacy_formats, "Legacy mode should create TIF files"

    def test_modern_and_legacy_produce_different_tile_counts(
        self,
        test_annotations_3_classes,
        test_image_list,
        test_modern_config,
        test_legacy_config,
        temp_model_directory
    ):
        """Test that modern and legacy modes produce different numbers of tiles.

        Note: Uses test-scaled configs (modern and legacy) instead of production
        configs to ensure tests complete quickly with 512x512 test images.
        """
        modern_dir = os.path.join(temp_model_directory, "modern")
        legacy_dir = os.path.join(temp_model_directory, "legacy")
        os.makedirs(modern_dir, exist_ok=True)
        os.makedirs(legacy_dir, exist_ok=True)

        # Create metadata files for both directories
        create_model_metadata(modern_dir)
        create_model_metadata(legacy_dir)

        # Run both modes with test-scaled configs
        create_training_tiles(
            model_path=modern_dir,
            annotations=test_annotations_3_classes,
            image_list=test_image_list,
            create_new_tiles=True,
            config=test_modern_config
        )

        create_training_tiles(
            model_path=legacy_dir,
            annotations=test_annotations_3_classes,
            image_list=test_image_list,
            create_new_tiles=True,
            config=test_legacy_config
        )

        # Count tiles (split tiles in training/im/, not big tiles)
        modern_tiles = list(Path(modern_dir).glob("training/im/*.*"))
        legacy_tiles = list(Path(legacy_dir).glob("training/im/*.*"))

        # Different reduction factors should produce different tile counts
        # (Test legacy uses reduction_factor=2, Modern uses reduction_factor=10)
        assert len(modern_tiles) != len(legacy_tiles), \
            "Modern and legacy should produce different tile counts due to different reduction factors"


@pytest.mark.integration
@pytest.mark.timeout(300)  # 5 minutes timeout for integration tests
class TestTileQualityMetrics:
    """Test tile quality metrics: black pixel ratio and tissue coverage."""

    def _calculate_black_pixel_ratio(self, image_path: str) -> float:
        """
        Calculate the ratio of black pixels in an image.

        Args:
            image_path: Path to image file

        Returns:
            Ratio of black pixels (0.0 to 1.0)
        """
        image = np.array(Image.open(image_path))
        if len(image.shape) == 3:
            # RGB image: consider pixel black if all channels are near zero
            black_mask = np.all(image < 10, axis=2)
        else:
            # Grayscale: consider pixel black if value is near zero
            black_mask = image < 10

        return np.mean(black_mask)

    def _calculate_tissue_coverage(self, image_path: str) -> float:
        """
        Calculate tissue coverage (non-white pixels) in an image.

        Args:
            image_path: Path to image file

        Returns:
            Ratio of tissue pixels (0.0 to 1.0)
        """
        image = np.array(Image.open(image_path))
        if len(image.shape) == 3:
            # RGB image: consider pixel tissue if not near white
            tissue_mask = np.any(image < 230, axis=2)
        else:
            # Grayscale: consider pixel tissue if not near white
            tissue_mask = image < 230

        return np.mean(tissue_mask)

    def test_modern_and_legacy_have_different_black_pixel_ratios(
        self,
        test_annotations_3_classes,
        test_image_list,
        test_modern_config,
        test_legacy_config,
        temp_model_directory
    ):
        """
        Test that modern and legacy modes produce different black pixel ratios.

        This test validates that the two tile generation modes produce measurably
        different outputs in terms of black pixel content. With test-scaled configs:
        - Modern mode (reduction_factor=10, big_tile_size=2048, no rotation cropping)
        - Legacy mode (reduction_factor=2, big_tile_size=1024, rotation cropping)

        Note: The direction of the difference depends on configuration parameters.
        With test configs, legacy mode typically has fewer black pixels due to
        finer placement (reduction_factor=2) and smaller big_tile_size. This test
        validates that the modes produce different behavior, not which is better.
        """
        modern_dir = os.path.join(temp_model_directory, "modern")
        legacy_dir = os.path.join(temp_model_directory, "legacy")
        os.makedirs(modern_dir, exist_ok=True)
        os.makedirs(legacy_dir, exist_ok=True)

        # Create metadata files for both directories
        create_model_metadata(modern_dir)
        create_model_metadata(legacy_dir)

        # Run both modes with test-specific scaled configurations
        create_training_tiles(
            model_path=modern_dir,
            annotations=test_annotations_3_classes,
            image_list=test_image_list,
            create_new_tiles=True,
            config=test_modern_config
        )

        create_training_tiles(
            model_path=legacy_dir,
            annotations=test_annotations_3_classes,
            image_list=test_image_list,
            create_new_tiles=True,
            config=test_legacy_config
        )

        # Calculate black pixel ratios
        modern_tiles = list(Path(modern_dir).rglob("HE*.*"))
        legacy_tiles = list(Path(legacy_dir).rglob("HE*.*"))

        modern_black_ratios = [self._calculate_black_pixel_ratio(str(t)) for t in modern_tiles]
        legacy_black_ratios = [self._calculate_black_pixel_ratio(str(t)) for t in legacy_tiles]

        # Average black pixel ratio
        modern_avg_black = np.mean(modern_black_ratios)
        legacy_avg_black = np.mean(legacy_black_ratios)

        # Modern and legacy should have significantly different black pixel ratios
        # (validates that the configuration differences produce measurable effects)
        # Use 5% threshold for "significantly different"
        assert abs(modern_avg_black - legacy_avg_black) > 0.05, \
            f"Modern and legacy modes should produce significantly different black pixel ratios " \
            f"(modern={modern_avg_black:.3f}, legacy={legacy_avg_black:.3f}, " \
            f"difference={abs(modern_avg_black - legacy_avg_black):.3f})"

    def test_tile_tissue_coverage(
        self,
        test_annotations_3_classes,
        test_image_list,
        test_modern_config,
        temp_model_directory
    ):
        """Test that tiles have reasonable tissue coverage."""
        modern_dir = os.path.join(temp_model_directory, "modern")
        os.makedirs(modern_dir, exist_ok=True)

        # Create metadata file
        create_model_metadata(modern_dir)

        create_training_tiles(
            model_path=modern_dir,
            annotations=test_annotations_3_classes,
            image_list=test_image_list,
            create_new_tiles=True,
            config=test_modern_config
        )

        # Calculate tissue coverage
        modern_tiles = list(Path(modern_dir).rglob("HE*.*"))
        tissue_coverages = [self._calculate_tissue_coverage(str(t)) for t in modern_tiles]

        # All tiles should have some tissue coverage
        assert all(tc > 0.0 for tc in tissue_coverages), \
            "All tiles should contain some tissue"

        # Average tissue coverage should be reasonable (>10%)
        avg_coverage = np.mean(tissue_coverages)
        assert avg_coverage > 0.1, \
            f"Average tissue coverage should be >10% (got {avg_coverage:.3f})"


@pytest.mark.integration
@pytest.mark.timeout(300)  # 5 minutes timeout for integration tests
class TestConfigurationOverride:
    """Test custom configuration and environment variable support."""

    def test_custom_config_overrides_defaults(
        self,
        test_annotations_3_classes,
        test_image_list,
        temp_model_directory
    ):
        """Test that custom config parameters are respected."""
        custom_config = TileGenerationConfig(
            mode="custom",
            reduction_factor=7,
            use_disk_filter=False,
            crop_rotations=False,
            class_rotation_frequency=4,
            deterministic_seed=42,
            big_tile_size=10000,
            file_format="jpg"
        )

        custom_dir = os.path.join(temp_model_directory, "custom")
        os.makedirs(custom_dir, exist_ok=True)

        # Create metadata file
        create_model_metadata(custom_dir)

        create_training_tiles(
            model_path=custom_dir,
            annotations=test_annotations_3_classes,
            image_list=test_image_list,
            create_new_tiles=True,
            config=custom_config
        )

        # Check that JPG files were created (custom format)
        custom_tiles = list(Path(custom_dir).rglob("HE*.*"))
        assert len(custom_tiles) > 0, "Custom config should create tiles"

        custom_formats = {f.suffix for f in custom_tiles}
        assert ".jpg" in custom_formats, "Custom config should create JPG files"

    def test_environment_variable_override(
        self,
        test_annotations_3_classes,
        test_image_list,
        temp_model_directory,
        monkeypatch
    ):
        """Test that environment variable overrides default mode."""
        # Set environment variable to legacy mode
        monkeypatch.setenv('CODAVISION_TILE_GENERATION_MODE', 'legacy')

        # Get default config (should be legacy due to env var)
        config = get_default_tile_config()
        assert config.mode == 'legacy'
        assert config.reduction_factor == 5
        assert config.file_format == 'tif'

        # Create tiles using default config (should use legacy)
        env_dir = os.path.join(temp_model_directory, "env_legacy")
        os.makedirs(env_dir, exist_ok=True)

        # Create metadata file
        create_model_metadata(env_dir)

        create_training_tiles(
            model_path=env_dir,
            annotations=test_annotations_3_classes,
            image_list=test_image_list,
            create_new_tiles=True
            # config=None uses get_default_tile_config()
        )

        # Check that TIFF files were created (legacy format)
        env_tiles = list(Path(env_dir).rglob("HE*.*"))
        env_formats = {f.suffix for f in env_tiles}
        assert ".tif" in env_formats, \
            "Environment variable should override to legacy mode (TIFF format)"


@pytest.mark.integration
@pytest.mark.timeout(300)  # 5 minutes timeout for integration tests
class TestDeterministicBehavior:
    """Test deterministic seeding and reproducibility."""

    def test_modern_mode_is_deterministic(
        self,
        test_annotations_3_classes,
        test_image_list,
        test_modern_config,
        temp_model_directory
    ):
        """Test that modern mode produces identical results across runs."""
        # Run modern mode twice
        run1_dir = os.path.join(temp_model_directory, "run1")
        run2_dir = os.path.join(temp_model_directory, "run2")
        os.makedirs(run1_dir, exist_ok=True)
        os.makedirs(run2_dir, exist_ok=True)

        # Create metadata files for both directories
        create_model_metadata(run1_dir)
        create_model_metadata(run2_dir)

        create_training_tiles(
            model_path=run1_dir,
            annotations=test_annotations_3_classes,
            image_list=test_image_list,
            create_new_tiles=True,
            config=test_modern_config
        )

        create_training_tiles(
            model_path=run2_dir,
            annotations=test_annotations_3_classes,
            image_list=test_image_list,
            create_new_tiles=True,
            config=test_modern_config
        )

        # Compare tile counts
        run1_tiles = sorted(Path(run1_dir).rglob("HE*.*"))
        run2_tiles = sorted(Path(run2_dir).rglob("HE*.*"))

        assert len(run1_tiles) == len(run2_tiles), \
            "Modern mode should produce identical tile counts across runs"

        # Compare first few tiles pixel-by-pixel (deterministic seed should produce identical results)
        for i in range(min(3, len(run1_tiles))):
            img1 = np.array(Image.open(run1_tiles[i]))
            img2 = np.array(Image.open(run2_tiles[i]))

            # Images should be identical
            assert np.array_equal(img1, img2), \
                f"Modern mode tiles should be identical across runs (tile {i})"



@pytest.mark.integration
@pytest.mark.timeout(60)  # 1 minute timeout for integration tests
class TestInputValidation:
    """Test input validation and error handling."""

    def test_missing_image_file_raises_error(
        self,
        test_annotations_3_classes,
        temp_model_directory
    ):
        """Test that missing image files raise FileNotFoundError."""
        broken_image_list = {
            "tile_name": ["nonexistent.png"],
            "tile_pth": [os.path.join(temp_model_directory, "im")]
        }

        create_model_metadata(temp_model_directory)

        with pytest.raises(FileNotFoundError):
            create_training_tiles_modern(
                model_path=temp_model_directory,
                annotations=test_annotations_3_classes,
                image_list=broken_image_list,
                create_new_tiles=True
            )

    def test_invalid_image_list_structure_raises_error(
        self,
        test_annotations_3_classes,
        temp_model_directory
    ):
        """Test that invalid image_list structure raises ValueError."""
        broken_image_list = {"wrong_key": ["test.png"]}

        create_model_metadata(temp_model_directory)

        with pytest.raises(ValueError):
            create_training_tiles_modern(
                model_path=temp_model_directory,
                annotations=test_annotations_3_classes,
                image_list=broken_image_list,
                create_new_tiles=True
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
