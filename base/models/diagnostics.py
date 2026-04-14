"""
Diagnostic instrumentation for the TF training loop.

This module provides a Keras callback that characterizes the mechanism
underlying validation-loss instability in the `lr_fix` configuration. It is
gated by the ``BN_DIAGNOSTICS=1`` environment variable and is inert otherwise,
so importing it has zero runtime cost on production training paths.

The primary callback, :class:`BNDiagnosticCallback`, logs per-validation:

- total gradient L2 norm on a fixed anchor batch
- BatchNormalization running mean/variance norms (summed across all BN layers)
- train/eval logit L2 distance on the anchor batch (direct measure of
  BN train/eval skew)
- logit and softmax statistics on the anchor batch in eval mode
- per-class cross-entropy loss on the anchor batch

On training start, it also writes a ``bn_layer_inventory.csv`` file listing
every BatchNormalization layer in the model with its trainable flag, momentum,
and epsilon, so we can verify that intervention commits (e.g. lowering BN
momentum) actually take effect where we think they do.
"""

from __future__ import annotations

import csv
import os
import time
from typing import List, Optional

import numpy as np
import tensorflow as tf
import keras

from base.utils.logger import Logger


def diagnostics_enabled() -> bool:
    """Return True if the BN diagnostic instrumentation is enabled via env var."""
    return os.environ.get("BN_DIAGNOSTICS", "").strip() in ("1", "true", "True", "yes", "on")


class BNDiagnosticCallback(keras.callbacks.Callback):
    """
    Per-validation diagnostic logger for BatchNorm lag hypothesis investigation.

    Writes one CSV row every ``log_every`` training batches. Computations are
    performed on a fixed anchor batch captured at training start, so every
    row is directly comparable across the training run. The callback also
    performs an auxiliary forward+backward pass on the anchor batch to capture
    the current gradient norm without modifying the training loop.
    """

    def __init__(
        self,
        val_dataset: tf.data.Dataset,
        loss_function: tf.keras.losses.Loss,
        num_classes: int,
        log_dir: str,
        log_every: int = 128,
        logger: Optional[Logger] = None,
    ):
        super().__init__()
        self.val_dataset = val_dataset
        self.loss_function = loss_function
        self.num_classes = int(num_classes)
        self.log_dir = log_dir
        self.log_every = int(log_every)
        self.logger = logger

        self._csv_path = os.path.join(log_dir, "bn_diagnostics.csv")
        self._inventory_path = os.path.join(log_dir, "bn_layer_inventory.csv")
        self._csv_file = None
        self._csv_writer = None

        self._anchor_x: Optional[tf.Tensor] = None
        self._anchor_y: Optional[tf.Tensor] = None
        self._bn_layers: List[keras.layers.BatchNormalization] = []
        self._global_step = 0
        self._start_wall = time.time()

    # ------------------------------------------------------------------ setup

    def _capture_anchor_batch(self) -> None:
        """Grab the first batch of the validation dataset and hold it in memory."""
        for x, y in self.val_dataset.take(1):
            # Copy to numpy then back so we detach from any dataset iterator state
            self._anchor_x = tf.convert_to_tensor(x.numpy())
            self._anchor_y = tf.convert_to_tensor(y.numpy())
            break
        if self._anchor_x is None:
            raise RuntimeError(
                "BNDiagnosticCallback: val_dataset produced zero batches; "
                "cannot capture an anchor batch."
            )

    def _enumerate_bn_layers(self) -> None:
        """Walk the model and record every BatchNormalization layer."""
        self._bn_layers = []
        for layer in self.model.layers:
            # keras.applications ResNet50 is typically a nested Model; recurse
            if isinstance(layer, keras.Model):
                for sub in layer.layers:
                    if isinstance(sub, keras.layers.BatchNormalization):
                        self._bn_layers.append(sub)
            elif isinstance(layer, keras.layers.BatchNormalization):
                self._bn_layers.append(layer)

    def _write_bn_inventory(self) -> None:
        """Emit a one-time CSV listing of all BN layers and their hyperparameters."""
        with open(self._inventory_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer_name", "trainable", "momentum", "epsilon", "num_features"])
            for bn in self._bn_layers:
                try:
                    num_features = int(bn.moving_mean.shape[-1]) if bn.moving_mean is not None else -1
                except Exception:
                    num_features = -1
                w.writerow([
                    bn.name,
                    bn.trainable,
                    float(getattr(bn, "momentum", -1.0)),
                    float(getattr(bn, "epsilon", -1.0)),
                    num_features,
                ])
        if self.logger:
            self.logger.logger.info(
                f"BNDiagnosticCallback: wrote {len(self._bn_layers)} BN layers to {self._inventory_path}"
            )

    def _open_csv(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        header = [
            "global_step",
            "epoch",
            "wall_time_s",
            "learning_rate",
            "grad_norm_global",
            "bn_running_mean_l2",
            "bn_running_var_l2",
            "train_eval_logit_l2",
            "anchor_logit_max",
            "anchor_logit_min",
            "anchor_logit_std",
            "anchor_softmax_max_mean",
            "anchor_total_loss",
        ] + [f"class_{c}_loss" for c in range(self.num_classes)]
        self._csv_writer.writerow(header)
        self._csv_file.flush()

    # ------------------------------------------------------------------ metrics

    def _bn_running_stat_norms(self) -> tuple[float, float]:
        """Sum L2 norms of all BN running_mean / running_variance across the model."""
        mean_sq = 0.0
        var_sq = 0.0
        for bn in self._bn_layers:
            if bn.moving_mean is not None:
                m = bn.moving_mean.numpy()
                mean_sq += float(np.sum(m * m))
            if bn.moving_variance is not None:
                v = bn.moving_variance.numpy()
                var_sq += float(np.sum(v * v))
        return float(np.sqrt(mean_sq)), float(np.sqrt(var_sq))

    def _compute_grad_norm(self) -> float:
        """Forward + backward on the anchor batch; return global gradient L2 norm."""
        with tf.GradientTape() as tape:
            logits = self.model(self._anchor_x, training=True)
            y_flat = tf.reshape(self._anchor_y, [-1])
            logits_flat = tf.reshape(logits, [-1, tf.shape(logits)[-1]])
            loss = self.loss_function(y_flat, logits_flat)
        grads = tape.gradient(loss, self.model.trainable_variables)
        sq = 0.0
        for g in grads:
            if g is None:
                continue
            sq += float(tf.reduce_sum(tf.square(g)).numpy())
        return float(np.sqrt(sq))

    def _compute_train_eval_divergence(self) -> tuple[float, dict]:
        """Forward anchor batch in train and eval modes; return logit L2 and eval stats."""
        logits_train = self.model(self._anchor_x, training=True)
        logits_eval = self.model(self._anchor_x, training=False)

        diff = logits_train - logits_eval
        l2 = float(tf.sqrt(tf.reduce_sum(tf.square(diff))).numpy())

        # Eval-mode stats
        eval_flat = tf.reshape(logits_eval, [-1, tf.shape(logits_eval)[-1]])
        logit_max = float(tf.reduce_max(eval_flat).numpy())
        logit_min = float(tf.reduce_min(eval_flat).numpy())
        logit_std = float(tf.math.reduce_std(eval_flat).numpy())

        softmax = tf.nn.softmax(eval_flat, axis=-1)
        softmax_max = tf.reduce_max(softmax, axis=-1)  # max prob per pixel
        softmax_max_mean = float(tf.reduce_mean(softmax_max).numpy())

        return l2, {
            "logit_max": logit_max,
            "logit_min": logit_min,
            "logit_std": logit_std,
            "softmax_max_mean": softmax_max_mean,
            "eval_flat": eval_flat,
        }

    def _per_class_losses(self, eval_flat: tf.Tensor) -> tuple[float, list]:
        """Compute per-class mean cross-entropy on the anchor batch in eval mode."""
        y_flat = tf.reshape(self._anchor_y, [-1])
        y_flat = tf.cast(y_flat, tf.int32)
        per_pixel = tf.keras.losses.sparse_categorical_crossentropy(
            y_flat, eval_flat, from_logits=True
        )
        per_class = []
        total = 0.0
        for c in range(self.num_classes):
            mask = tf.cast(tf.equal(y_flat, c), tf.float32)
            n = tf.reduce_sum(mask)
            if float(n.numpy()) > 0:
                mean_c = float((tf.reduce_sum(per_pixel * mask) / n).numpy())
            else:
                mean_c = float("nan")
            per_class.append(mean_c)
            if not np.isnan(mean_c):
                total += mean_c
        return total, per_class

    def _current_lr(self) -> float:
        try:
            lr = self.model.optimizer.learning_rate
            if hasattr(lr, "numpy"):
                return float(lr.numpy())
            return float(lr)
        except Exception:
            return float("nan")

    # ------------------------------------------------------------------ hooks

    def on_train_begin(self, logs=None):
        self._capture_anchor_batch()
        self._enumerate_bn_layers()
        self._write_bn_inventory()
        self._open_csv()
        # Log the initial (pre-training) tick so we have a baseline row
        self._write_row(epoch=0)

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        self._global_step += 1
        if self._global_step % self.log_every == 0:
            self._write_row(epoch=getattr(self, "_current_epoch", -1))

    def on_train_end(self, logs=None):
        # Final tick at end of training
        self._write_row(epoch=getattr(self, "_current_epoch", -1))
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None

    # ------------------------------------------------------------------ writer

    def _write_row(self, epoch: int) -> None:
        try:
            grad_norm = self._compute_grad_norm()
            mean_l2, var_l2 = self._bn_running_stat_norms()
            train_eval_l2, eval_stats = self._compute_train_eval_divergence()
            total_loss, per_class = self._per_class_losses(eval_stats["eval_flat"])
            lr = self._current_lr()

            row = [
                self._global_step,
                epoch,
                round(time.time() - self._start_wall, 2),
                lr,
                grad_norm,
                mean_l2,
                var_l2,
                train_eval_l2,
                eval_stats["logit_max"],
                eval_stats["logit_min"],
                eval_stats["logit_std"],
                eval_stats["softmax_max_mean"],
                total_loss,
            ] + per_class
            self._csv_writer.writerow(row)
            self._csv_file.flush()
        except Exception as exc:
            if self.logger:
                self.logger.logger.warning(
                    f"BNDiagnosticCallback: row write failed at step {self._global_step}: {exc}"
                )
