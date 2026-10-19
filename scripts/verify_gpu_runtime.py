"""Exercise both frameworks and CODAvision's AdamW paths in one GPU process.

Run from an activated environment: uv run --locked python scripts/verify_gpu_runtime.py
CPU fallback is a failure. No external dataset or model download is needed.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import numpy as np
    import tensorflow as tf
    import torch

    print("TensorFlow:", tf.__version__, "build:", tf.sysconfig.get_build_info(), flush=True)
    print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, flush=True)
    gpus = tf.config.list_physical_devices("GPU")
    assert gpus, "TensorFlow cannot see a GPU; check the legacy Windows CUDA runtime."
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_soft_device_placement(False)
    assert torch.cuda.is_available(), "PyTorch cannot use CUDA."
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    from base.models.training import DeepLabV3PlusTrainer, UNetTrainer

    for trainer_class in (DeepLabV3PlusTrainer, UNetTrainer):
        # Exercise the real compilation path without requiring a training dataset.
        trainer = trainer_class.__new__(trainer_class)
        trainer.use_adamw_optimizer = True
        trainer.learning_rate = 0.001
        trainer.l2_regularization_weight = 0.01
        trainer.optimizer_epsilon = 1e-8
        trainer.loss_function = tf.keras.losses.MeanSquaredError()
        with tf.device("/GPU:0"):
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(16, 16, 3)),
                tf.keras.layers.Conv2D(2, 3, kernel_initializer="ones"),
            ])
            trainer._compile_model(model)
            before = model.trainable_variables[0].numpy().copy()
            with tf.GradientTape() as tape:
                output = model(tf.ones((1, 16, 16, 3)), training=True)
                loss = tf.reduce_mean(output ** 2)
            gradients = tape.gradient(loss, model.trainable_variables)
            assert "GPU" in output.device.upper(), output.device
            for gradient in gradients:
                assert gradient is not None
                tf.debugging.assert_all_finite(gradient, "Invalid TensorFlow gradient")
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        assert np.isfinite(model.trainable_variables[0].numpy()).all()
        assert not np.array_equal(before, model.trainable_variables[0].numpy())
        # Keras graph-mode training must work too, not just eager apply_gradients.
        metrics = model.train_on_batch(
            np.ones((1, 16, 16, 3), dtype=np.float32),
            np.zeros((1, 14, 14, 2), dtype=np.float32),
        )
        assert np.isfinite(metrics).all(), metrics
        print(f"TF GPU forward/backward PASS: {trainer_class.__name__}, "
              f"jit_compile={model.optimizer.jit_compile}, {output.device}", flush=True)

    model = torch.nn.Conv2d(3, 2, 3).cuda()
    output = model(torch.ones((1, 3, 16, 16), device="cuda"))
    loss = output.square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert output.is_cuda
    assert model.weight.grad is not None
    assert torch.isfinite(model.weight.grad).all().item()
    print("Torch GPU forward/backward PASS:", output.device, loss.item(), flush=True)
    print("PASS: both frameworks executed GPU operations in the same process.", flush=True)


if __name__ == "__main__":
    main()
