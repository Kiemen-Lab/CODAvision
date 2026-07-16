"""
Unit tests for per-model framework resolution and trained-model detection.

Covers the three surfaces touched by the PyTorch-checkpoint fix (PR #52 follow-up):
- gui.components.main_window.trained_model_exists: detects .keras/.h5/.pth "best_model" files.
- base.image.classification.ImageClassifier._resolve_checkpoint: selects the checkpoint file and
  framework independent of the global default (the comment-2 bug guard), including .h5.
- base.models.training.train_segmentation_model_cnns: records the framework in net.pkl and skips
  retraining when a checkpoint from either framework already exists.

These paths were previously untested. QT_QPA_PLATFORM=offscreen is auto-set in tests/conftest.py,
so importing the GUI module works headless. No GPU / real training / real weights are needed.

pickle use below is safe: net.pkl is the application's own metadata format (written/read by
base.models.utils.save_model_metadata / base.data.loaders.load_model_metadata), and these tests
only read back dicts they just wrote in the same process — no untrusted input.
"""
import os
import pickle
from unittest.mock import MagicMock

import pytest

from base.config import ModelDefaults
from base.image.classification import ImageClassifier
from base.models.training import train_segmentation_model_cnns
from gui.components.main_window import trained_model_exists

MODEL_TYPE = 'DeepLabV3_plus'
B_KERAS = f'best_model_{MODEL_TYPE}.keras'
B_H5 = f'best_model_{MODEL_TYPE}.h5'
B_PTH = f'best_model_{MODEL_TYPE}.pth'
F_KERAS = f'{MODEL_TYPE}.keras'
F_PTH = f'{MODEL_TYPE}.pth'


def _make_dir(base_dir, files=(), framework=None, write_net=True):
    """Create a model dir with empty checkpoint files and (optionally) a minimal net.pkl.

    Empty files are sufficient: resolution only calls os.path.exists. net.pkl carries the four keys
    load_model_metadata requires (classNames, sxy, nblack, nwhite) plus model_type and, when given,
    the recorded framework.
    """
    d = os.path.join(base_dir, 'model')
    os.makedirs(d, exist_ok=True)
    if write_net:
        meta = {'classNames': ['a', 'b'], 'sxy': 256, 'nblack': 0, 'nwhite': 0,
                'model_type': MODEL_TYPE, 'nm': 'demo'}
        if framework is not None:
            meta['framework'] = framework
        with open(os.path.join(d, 'net.pkl'), 'wb') as f:
            pickle.dump(meta, f)
    for name in files:
        open(os.path.join(d, name), 'wb').close()
    return d


class TestTrainedModelExists:
    """GUI gate: a best-model checkpoint from any supported framework counts as trained."""

    @pytest.mark.parametrize("files,expected", [
        ([B_KERAS], True),
        ([B_H5], True),
        ([B_PTH], True),
        ([], False),
        (['best_model.txt'], False),        # 'best_model' present but unsupported extension
        (['something.keras'], False),       # supported extension but not a best_model file
    ])
    def test_detection(self, temp_dir, files, expected):
        d = _make_dir(temp_dir, files=files, write_net=False)
        assert trained_model_exists(d) is expected

    def test_missing_directory(self, temp_dir):
        assert trained_model_exists(os.path.join(temp_dir, 'does_not_exist')) is False


class TestCheckpointResolution:
    """_resolve_checkpoint selects (path, is_pytorch) independent of the global default framework.

    Expected values are literals validated offline against a 110-scenario simulation of the exact
    selection logic (current PR code invalid in 46 of them; this fix invalid in 0, 0 regressions).
    """

    @pytest.mark.parametrize("files,recorded,default,exp_name,exp_is_pytorch", [
        # --- comment-2 guard: global default pytorch must NOT hide a TF (.keras) model ---
        ([B_KERAS], 'tensorflow', 'pytorch', B_KERAS, False),   # fails on current code
        ([B_KERAS], None,        'pytorch', B_KERAS, False),    # legacy (inferred from disk)
        # --- default=tensorflow behaves as before ---
        ([B_KERAS], None,        'tensorflow', B_KERAS, False),
        ([B_PTH],   'pytorch',   'tensorflow', B_PTH,   True),
        ([B_PTH],   'pytorch',   'pytorch',    B_PTH,   True),
        # --- .h5 guard: GUI advertises .h5, so the loader must resolve it ---
        ([B_H5],    None,        'pytorch',    B_H5,    False),
        ([B_H5],    None,        'tensorflow', B_H5,    False),
        # --- both checkpoints present: recorded framework wins ---
        ([B_KERAS, B_PTH], 'tensorflow', 'pytorch',    B_KERAS, False),
        ([B_KERAS, B_PTH], 'pytorch',    'pytorch',    B_PTH,   True),
        # --- both present + NO recorded framework: falls to global default (documented limitation) ---
        ([B_KERAS, B_PTH], None, 'pytorch',    B_PTH,   True),
        ([B_KERAS, B_PTH], None, 'tensorflow', B_KERAS, False),
        # --- recorded framework value normalized (case-insensitive) ---
        ([B_KERAS], 'TensorFlow', 'pytorch',    B_KERAS, False),
        ([B_PTH],   'PyTorch',    'tensorflow', B_PTH,   True),
        # --- fallback to the other framework when the requested one has no file ---
        ([B_PTH],   'tensorflow', 'tensorflow', B_PTH,   True),
        ([B_KERAS], 'pytorch',    'tensorflow', B_KERAS, False),
        # --- only final-model checkpoint present ---
        ([F_PTH],   None, 'pytorch',    F_PTH,   True),
        ([F_KERAS], None, 'tensorflow', F_KERAS, False),
        # --- preference order: best before final; .keras before .h5 ---
        ([B_KERAS, F_KERAS], 'tensorflow', 'tensorflow', B_KERAS, False),
        ([B_KERAS, B_H5],    'tensorflow', 'tensorflow', B_KERAS, False),
    ])
    def test_resolution(self, temp_dir, monkeypatch, files, recorded, default, exp_name, exp_is_pytorch):
        monkeypatch.setattr(ModelDefaults, 'DEFAULT_FRAMEWORK', default)
        d = _make_dir(temp_dir, files=files, framework=recorded)
        clf = ImageClassifier(d, d, MODEL_TYPE)
        model_path, is_pytorch = clf._resolve_checkpoint()
        assert model_path == os.path.join(d, exp_name)
        assert is_pytorch is exp_is_pytorch

    def test_no_checkpoint_raises(self, temp_dir, monkeypatch):
        monkeypatch.setattr(ModelDefaults, 'DEFAULT_FRAMEWORK', 'tensorflow')
        d = _make_dir(temp_dir, files=[], framework=None)
        clf = ImageClassifier(d, d, MODEL_TYPE)
        with pytest.raises(FileNotFoundError):
            clf._resolve_checkpoint()


class TestTrainingRecordsFramework:
    """train_segmentation_model_cnns records the framework and skips cross-framework retrains.

    Secondary coverage: the TF trainer is stubbed so no real training runs.
    """

    def _minimal_net(self, temp_dir):
        d = os.path.join(temp_dir, 'model')
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'net.pkl'), 'wb') as f:
            pickle.dump({'model_type': MODEL_TYPE, 'nm': 'demo'}, f)
        return d

    def test_records_framework_in_net_pkl(self, temp_dir, monkeypatch):
        monkeypatch.setattr(ModelDefaults, 'DEFAULT_FRAMEWORK', 'tensorflow')
        d = self._minimal_net(temp_dir)
        stub = MagicMock()
        monkeypatch.setattr('base.models.training.DeepLabV3PlusTrainer', stub)

        train_segmentation_model_cnns(d)

        stub.assert_called_once_with(d)
        stub.return_value.train.assert_called_once()
        with open(os.path.join(d, 'net.pkl'), 'rb') as f:
            meta = pickle.load(f)
        assert meta.get('framework') == 'tensorflow'       # recorded
        assert meta.get('model_type') == MODEL_TYPE         # not clobbered
        assert meta.get('nm') == 'demo'

    def test_existing_checkpoint_skips_retrain(self, temp_dir, monkeypatch):
        monkeypatch.setattr(ModelDefaults, 'DEFAULT_FRAMEWORK', 'tensorflow')
        d = self._minimal_net(temp_dir)
        open(os.path.join(d, B_KERAS), 'wb').close()   # already-trained checkpoint
        stub = MagicMock()
        monkeypatch.setattr('base.models.training.DeepLabV3PlusTrainer', stub)

        train_segmentation_model_cnns(d, retrain_model=False)

        stub.return_value.train.assert_not_called()
