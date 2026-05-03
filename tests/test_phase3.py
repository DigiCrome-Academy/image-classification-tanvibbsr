"""
Phase 3 autograder — Transfer Learning.

Grading rubric (total 25 pts):
    [5 pts] build_transfer_model loads a valid pretrained base
    [5 pts] Base model is frozen when freeze_base=True
    [5 pts] Custom classification head has correct structure
    [5 pts] unfreeze_top_layers unfreezes the right number of layers
    [5 pts] Fine-tuned model uses a lower learning rate
"""

import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras


SUPPORTED_BASES = ["MobileNetV2", "VGG16", "ResNet50"]


def _build_transfer():
    try:
        from src.models import build_transfer_model
        return build_transfer_model
    except ImportError as e:
        pytest.skip(f"src.models not importable: {e}")


def _unfreeze():
    try:
        from src.models import unfreeze_top_layers
        return unfreeze_top_layers
    except ImportError as e:
        pytest.skip(f"src.models not importable: {e}")


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestTransferModel:

    @pytest.mark.parametrize("base", SUPPORTED_BASES)
    def test_builds_each_base(self, base):
        """Each supported base model name should build without error."""
        build = _build_transfer()
        model = build(base_model_name=base, input_shape=(64, 64, 3))
        assert isinstance(model, keras.Model), (
            f"build_transfer_model('{base}') must return a keras.Model"
        )

    def test_output_shape(self):
        build = _build_transfer()
        model = build(input_shape=(64, 64, 3))
        assert model.output_shape == (None, 1), (
            f"Expected output shape (None, 1), got {model.output_shape}"
        )

    def test_base_frozen_by_default(self):
        """All layers in the base model should be non-trainable when freeze_base=True."""
        build = _build_transfer()
        model = build(input_shape=(64, 64, 3), freeze_base=True)
        # The base model is a sub-layer; find it by counting params
        base_candidates = [
            l for l in model.layers
            if hasattr(l, "layers") and len(getattr(l, "layers", [])) > 5
        ]
        assert len(base_candidates) >= 1, (
            "Could not identify the base model sub-layer inside the transfer model"
        )
        base = base_candidates[0]
        trainable_in_base = [l for l in base.layers if l.trainable]
        assert len(trainable_in_base) == 0, (
            f"freeze_base=True: expected 0 trainable base layers, "
            f"found {len(trainable_in_base)}"
        )

    def test_base_unfrozen_when_requested(self):
        """freeze_base=False should leave base layers trainable."""
        build = _build_transfer()
        model = build(input_shape=(64, 64, 3), freeze_base=False)
        base_candidates = [
            l for l in model.layers
            if hasattr(l, "layers") and len(getattr(l, "layers", [])) > 5
        ]
        assert len(base_candidates) >= 1
        base = base_candidates[0]
        trainable_in_base = [l for l in base.layers if l.trainable]
        assert len(trainable_in_base) > 0, (
            "freeze_base=False: at least some base layers should be trainable"
        )

    def test_has_global_average_pooling(self):
        """Classification head must use GlobalAveragePooling2D."""
        build = _build_transfer()
        model = build(input_shape=(64, 64, 3))
        gap_layers = [l for l in model.layers
                      if isinstance(l, keras.layers.GlobalAveragePooling2D)]
        assert len(gap_layers) >= 1, (
            "Transfer model head must include a GlobalAveragePooling2D layer"
        )

    def test_has_dropout_in_head(self):
        build = _build_transfer()
        model = build(input_shape=(64, 64, 3), dropout_rate=0.3)
        dropout_layers = [l for l in model.layers
                          if isinstance(l, keras.layers.Dropout)]
        assert len(dropout_layers) >= 1, (
            "Classification head must include a Dropout layer"
        )

    def test_has_auc_metric(self):
        build = _build_transfer()
        model = build(input_shape=(64, 64, 3))
        metric_names = [m.name.lower() for m in model.metrics]
        assert any("auc" in n for n in metric_names), (
            f"Transfer model must include AUC metric, found: {metric_names}"
        )

    def test_predictions_in_range(self):
        build = _build_transfer()
        model = build(input_shape=(64, 64, 3))
        dummy = np.random.rand(2, 64, 64, 3).astype("float32")
        preds = model.predict(dummy, verbose=0)
        assert ((preds >= 0) & (preds <= 1)).all()


class TestUnfreeze:

    def test_unfreeze_top_layers_callable(self):
        fn = _unfreeze()
        assert callable(fn)

    def test_unfreezes_layers(self):
        """After unfreeze_top_layers, some base layers must become trainable."""
        build = _build_transfer()
        unfreeze = _unfreeze()
        model = build(input_shape=(64, 64, 3), freeze_base=True)
        model = unfreeze(model, num_layers=10)
        base_candidates = [
            l for l in model.layers
            if hasattr(l, "layers") and len(getattr(l, "layers", [])) > 5
        ]
        assert len(base_candidates) >= 1
        base = base_candidates[0]
        trainable_count = sum(1 for l in base.layers if l.trainable)
        assert trainable_count > 0, (
            "unfreeze_top_layers must make at least some base layers trainable"
        )

    def test_returns_compiled_model(self):
        build = _build_transfer()
        unfreeze = _unfreeze()
        model = build(input_shape=(64, 64, 3), freeze_base=True)
        model = unfreeze(model, num_layers=5)
        assert model.optimizer is not None, (
            "unfreeze_top_layers must recompile the model"
        )

    def test_fine_tune_lr_lower(self):
        """Fine-tuning should use a learning rate <= 1e-4."""
        build = _build_transfer()
        unfreeze = _unfreeze()
        model = build(input_shape=(64, 64, 3), learning_rate=1e-3, freeze_base=True)
        model = unfreeze(model, num_layers=10)
        lr = float(model.optimizer.learning_rate)
        assert lr <= 1e-4, (
            f"Fine-tuning learning rate should be <= 1e-4, got {lr}"
        )
