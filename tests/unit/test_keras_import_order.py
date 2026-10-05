"""Exercise Keras import order in fresh interpreters, independent of collection."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("tensorflow_first", [True, False])
def test_backbone_uses_legacy_keras_in_either_import_order(tensorflow_first, tmp_path):
    imports = (
        "import tensorflow as tf\nfrom tensorflow import keras\nimport base\n"
        if tensorflow_first
        else "import base\nimport tensorflow as tf\nfrom tensorflow import keras\n"
    )
    script = imports + """
import numpy as np
import sys
from base.models import backbones_tf
from base.models.training import RegularizationLossCallback

# Use the real backbone block without downloading pretrained encoder weights.
builder = backbones_tf.DeepLabV3Plus(32, 2, l2_regularization_weight=1e-4)
inputs = keras.Input(shape=(8, 8, 3))
outputs = builder.convolution_block(inputs, num_filters=2)
model = backbones_tf.Model(inputs, outputs)
assert isinstance(model, keras.Model)
assert backbones_tf.applications.ResNet50 is keras.applications.ResNet50
assert model.losses
callback = RegularizationLossCallback()
callback.set_model(model)
sample = np.ones((1, 8, 8, 3), dtype=np.float32)
expected = model(sample, training=False).numpy()
path = sys.argv[1]
model.save(path, save_format="h5")
restored = keras.models.load_model(path, compile=False)
np.testing.assert_allclose(restored(sample, training=False).numpy(), expected)
"""
    env = os.environ.copy()
    # A preconfigured parent or another test must not hide the import-order bug.
    env.pop("TF_USE_LEGACY_KERAS", None)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "model.keras")],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
