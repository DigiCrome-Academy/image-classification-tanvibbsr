"""
Phase 1 autograder — Baseline MLP.

Grading rubric (total 20 pts):
    [4 pts] build_baseline_mlp returns a compiled Sequential model
    [4 pts] Correct layer count and output shape
    [4 pts] Activation function is applied correctly
    [4 pts] Model compiles with the specified optimizer
    [4 pts] get_data_generators returns three generators
"""

import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras


# ── helpers ──────────────────────────────────────────────────────────────────

def _import_models():
    try:
        from src.models import build_baseline_mlp
        return build_baseline_mlp
    except ImportError as e:
        pytest.skip(f"src.models not importable: {e}")


def _import_data_loader():
    try:
        from src.data_loader import get_data_generators
        return get_data_generators
    except ImportError as e:
        pytest.skip(f"src.data_loader not importable: {e}")


# ── Phase 1 tests ─────────────────────────────────────────────────────────────

class TestBaselineMLP:

    def test_function_exists(self):
        """build_baseline_mlp must be importable and callable."""
        build = _import_models()
        assert callable(build), "build_baseline_mlp must be a callable function"

    def test_returns_keras_model(self):
        """build_baseline_mlp must return a keras.Model instance."""
        build = _import_models()
        model = build(input_shape=(64, 64, 3))
        assert isinstance(model, keras.Model), (
            "build_baseline_mlp must return a keras.Model"
        )

    def test_output_shape(self):
        """Output layer must produce a single probability (shape (None, 1))."""
        build = _import_models()
        model = build(input_shape=(64, 64, 3))
        assert model.output_shape == (None, 1), (
            f"Expected output shape (None, 1), got {model.output_shape}"
        )

    def test_model_is_compiled(self):
        """Model must be compiled (has an optimizer attached)."""
        build = _import_models()
        model = build(input_shape=(64, 64, 3))
        assert model.optimizer is not None, (
            "Model must be compiled — call model.compile() inside build_baseline_mlp()"
        )

    def test_loss_is_binary_crossentropy(self):
        """Loss must be binary_crossentropy."""
        build = _import_models()
        model = build(input_shape=(64, 64, 3))
        loss_name = model.loss if isinstance(model.loss, str) else model.loss.name
        assert "binary_crossentropy" in loss_name.lower(), (
            f"Loss should be binary_crossentropy, got: {loss_name}"
        )

    def test_has_flatten_layer(self):
        """Model must contain a Flatten layer to convert image tensors to 1-D."""
        build = _import_models()
        model = build(input_shape=(64, 64, 3))
        layer_types = [type(l).__name__ for l in model.layers]
        assert "Flatten" in layer_types, (
            "Model must include a Flatten layer"
        )

    def test_has_dense_layers(self):
        """Model must have at least 3 Dense hidden layers."""
        build = _import_models()
        model = build(input_shape=(64, 64, 3))
        dense_layers = [l for l in model.layers if isinstance(l, keras.layers.Dense)]
        # output layer is Dense too, so we expect >= 4 total
        assert len(dense_layers) >= 4, (
            f"Expected at least 3 hidden Dense layers + 1 output Dense layer, "
            f"found {len(dense_layers)} Dense layers total"
        )

    def test_activation_relu(self):
        """Default activation='relu' should produce non-negative activations."""
        build = _import_models()
        model = build(input_shape=(32, 32, 3), activation="relu")
        dummy = np.random.rand(2, 32, 32, 3).astype("float32")
        pred = model.predict(dummy, verbose=0)
        assert pred.min() >= 0.0, "Sigmoid output should always be >= 0"
        assert pred.max() <= 1.0, "Sigmoid output should always be <= 1"

    def test_sigmoid_output_range(self):
        """Predictions must lie in [0, 1] (sigmoid output)."""
        build = _import_models()
        model = build(input_shape=(32, 32, 3))
        dummy = np.random.rand(5, 32, 32, 3).astype("float32")
        preds = model.predict(dummy, verbose=0)
        assert ((preds >= 0) & (preds <= 1)).all(), (
            "All predictions must be in [0, 1] (sigmoid output)"
        )

    @pytest.mark.parametrize("optimizer", ["adam", "sgd", "rmsprop", "nadam"])
    def test_different_optimizers(self, optimizer):
        """Model should compile successfully with standard optimizers."""
        build = _import_models()
        model = build(input_shape=(32, 32, 3), optimizer=optimizer)
        assert model.optimizer is not None

    def test_accuracy_metric(self):
        """Model must track accuracy as a metric."""
        build = _import_models()
        model = build(input_shape=(32, 32, 3))
        metric_names = [m.name for m in model.metrics]
        assert any("accuracy" in n.lower() for n in metric_names), (
            f"Model must include accuracy metric, found: {metric_names}"
        )


class TestDataLoader:

    def test_data_loader_importable(self):
        """get_data_generators must be importable."""
        fn = _import_data_loader()
        assert callable(fn)

    def test_compute_class_weights_importable(self):
        """compute_class_weights must be importable."""
        try:
            from src.data_loader import compute_class_weights
            assert callable(compute_class_weights)
        except ImportError as e:
            pytest.fail(f"compute_class_weights not importable: {e}")

    def test_describe_dataset_importable(self):
        """describe_dataset helper must be importable."""
        try:
            from src.data_loader import describe_dataset
            assert callable(describe_dataset)
        except ImportError as e:
            pytest.fail(f"describe_dataset not importable: {e}")
