"""
Phase 2 autograder — Advanced Neural Network.

Grading rubric (total 20 pts):
    [5 pts] build_advanced_nn returns a compiled model with BN + Dropout
    [5 pts] L2 regularisation applied to Dense layers
    [5 pts] Gradient clipping is set on the optimizer
    [5 pts] get_callbacks returns correct callback types
"""

import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras


def _build():
    try:
        from src.models import build_advanced_nn
        return build_advanced_nn
    except ImportError as e:
        pytest.skip(f"src.models not importable: {e}")


def _callbacks():
    try:
        from src.train import get_callbacks
        return get_callbacks
    except ImportError as e:
        pytest.skip(f"src.train not importable: {e}")


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestAdvancedNN:

    def test_returns_keras_model(self):
        build = _build()
        model = build(input_shape=(64, 64, 3))
        assert isinstance(model, keras.Model)

    def test_output_shape(self):
        build = _build()
        model = build(input_shape=(64, 64, 3))
        assert model.output_shape == (None, 1), (
            f"Expected (None, 1), got {model.output_shape}"
        )

    def test_has_batch_normalization(self):
        """At least one BatchNormalization layer required."""
        build = _build()
        model = build(input_shape=(64, 64, 3))
        bn_layers = [l for l in model.layers
                     if isinstance(l, keras.layers.BatchNormalization)]
        assert len(bn_layers) >= 1, (
            "build_advanced_nn must include at least one BatchNormalization layer"
        )

    def test_has_dropout(self):
        """At least one Dropout layer required."""
        build = _build()
        model = build(input_shape=(64, 64, 3))
        dropout_layers = [l for l in model.layers
                          if isinstance(l, keras.layers.Dropout)]
        assert len(dropout_layers) >= 1, (
            "build_advanced_nn must include at least one Dropout layer"
        )

    def test_dropout_rate(self):
        """Dropout rate should be applied as specified."""
        build = _build()
        model = build(input_shape=(64, 64, 3), dropout_rate=0.4)
        dropout_layers = [l for l in model.layers
                          if isinstance(l, keras.layers.Dropout)]
        assert len(dropout_layers) >= 1
        rates = [l.rate for l in dropout_layers]
        assert any(abs(r - 0.4) < 1e-6 for r in rates), (
            f"Expected a Dropout layer with rate=0.4, found rates: {rates}"
        )

    def test_l2_regularization(self):
        """Dense layers must use L2 kernel regularisation."""
        build = _build()
        model = build(input_shape=(64, 64, 3), l2_lambda=1e-4)
        dense_with_reg = [
            l for l in model.layers
            if isinstance(l, keras.layers.Dense)
            and l.kernel_regularizer is not None
        ]
        assert len(dense_with_reg) >= 1, (
            "At least one Dense layer must have kernel_regularizer set (L2)"
        )

    def test_gradient_clipping(self):
        """Optimizer must have clipnorm or clipvalue set."""
        build = _build()
        model = build(input_shape=(64, 64, 3))
        opt = model.optimizer
        has_clip = (
            getattr(opt, "clipnorm", None) is not None
            or getattr(opt, "clipvalue", None) is not None
        )
        assert has_clip, (
            "Optimizer must have gradient clipping (clipnorm or clipvalue)"
        )

    def test_has_auc_metric(self):
        """Model must track AUC as a metric."""
        build = _build()
        model = build(input_shape=(64, 64, 3))
        metric_names = [m.name.lower() for m in model.metrics]
        assert any("auc" in n for n in metric_names), (
            f"Model must include AUC metric, found: {metric_names}"
        )

    def test_predictions_in_range(self):
        build = _build()
        model = build(input_shape=(32, 32, 3))
        dummy = np.random.rand(4, 32, 32, 3).astype("float32")
        preds = model.predict(dummy, verbose=0)
        assert ((preds >= 0) & (preds <= 1)).all()

    def test_functional_api(self):
        """Model should be built with the Functional API (not Sequential)."""
        build = _build()
        model = build(input_shape=(64, 64, 3))
        # Sequential models are a subclass; Functional models are not
        assert not isinstance(model, keras.Sequential), (
            "build_advanced_nn should use the Functional API, not Sequential"
        )


class TestCallbacks:

    def test_returns_list(self):
        get_cb = _callbacks()
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            cbs = get_cb(model_name="test", checkpoint_dir=pathlib.Path(tmp))
        assert isinstance(cbs, list), "get_callbacks must return a list"

    def test_has_early_stopping(self):
        get_cb = _callbacks()
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            cbs = get_cb(model_name="test", checkpoint_dir=pathlib.Path(tmp))
        types = [type(cb).__name__ for cb in cbs]
        assert "EarlyStopping" in types, (
            f"get_callbacks must include EarlyStopping, found: {types}"
        )

    def test_has_model_checkpoint(self):
        get_cb = _callbacks()
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            cbs = get_cb(model_name="test", checkpoint_dir=pathlib.Path(tmp))
        types = [type(cb).__name__ for cb in cbs]
        assert "ModelCheckpoint" in types, (
            f"get_callbacks must include ModelCheckpoint, found: {types}"
        )

    def test_has_reduce_lr(self):
        get_cb = _callbacks()
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            cbs = get_cb(model_name="test", checkpoint_dir=pathlib.Path(tmp))
        types = [type(cb).__name__ for cb in cbs]
        assert "ReduceLROnPlateau" in types, (
            f"get_callbacks must include ReduceLROnPlateau, found: {types}"
        )

    def test_early_stopping_restores_weights(self):
        get_cb = _callbacks()
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            cbs = get_cb(model_name="test", checkpoint_dir=pathlib.Path(tmp))
        es = next(cb for cb in cbs if isinstance(cb, keras.callbacks.EarlyStopping))
        assert es.restore_best_weights, (
            "EarlyStopping must have restore_best_weights=True"
        )
