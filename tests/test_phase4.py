"""
Phase 4 autograder — Optimisation, Ensemble & Evaluation.

Grading rubric (total 35 pts — covers 10% Model Analysis + 25% remaining):
    [7 pts]  build_ensemble returns a model that averages predictions
    [7 pts]  evaluate_model returns a dict with required keys
    [7 pts]  plot_confusion_matrix runs without error
    [7 pts]  plot_roc_curve runs without error
    [7 pts]  grad_cam returns a 2-D heatmap in [0, 1]
"""

import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for CI


# ── helpers ───────────────────────────────────────────────────────────────────

def _tiny_binary_model(input_shape=(32, 32, 3)):
    """Tiny trained model for testing evaluation/ensemble utilities."""
    inp = keras.Input(shape=input_shape)
    x = keras.layers.Flatten()(inp)
    x = keras.layers.Dense(8, activation="relu")(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _fake_test_data(n=20, img_shape=(32, 32, 3)):
    X = np.random.rand(n, *img_shape).astype("float32")
    y = np.random.randint(0, 2, size=(n,)).astype("float32")
    ds = tf.data.Dataset.from_tensor_slices((X, y)).batch(8)
    return ds, y


# ── Ensemble ──────────────────────────────────────────────────────────────────

class TestEnsemble:

    def test_ensemble_importable(self):
        try:
            from src.models import build_ensemble
            assert callable(build_ensemble)
        except ImportError as e:
            pytest.fail(f"build_ensemble not importable: {e}")

    def test_returns_keras_model(self):
        from src.models import build_ensemble
        m1 = _tiny_binary_model()
        m2 = _tiny_binary_model()
        ensemble = build_ensemble([m1, m2], input_shape=(32, 32, 3))
        assert isinstance(ensemble, keras.Model)

    def test_output_shape(self):
        from src.models import build_ensemble
        m1 = _tiny_binary_model()
        m2 = _tiny_binary_model()
        ensemble = build_ensemble([m1, m2], input_shape=(32, 32, 3))
        assert ensemble.output_shape == (None, 1), (
            f"Ensemble output should be (None, 1), got {ensemble.output_shape}"
        )

    def test_predictions_are_averaged(self):
        """Ensemble predictions should differ from any single model's predictions."""
        from src.models import build_ensemble
        m1 = _tiny_binary_model()
        m2 = _tiny_binary_model()
        ensemble = build_ensemble([m1, m2], input_shape=(32, 32, 3))
        dummy = np.random.rand(4, 32, 32, 3).astype("float32")
        p1 = m1.predict(dummy, verbose=0)
        p2 = m2.predict(dummy, verbose=0)
        pe = ensemble.predict(dummy, verbose=0)
        expected_avg = (p1 + p2) / 2.0
        np.testing.assert_allclose(pe, expected_avg, atol=1e-5, err_msg=(
            "Ensemble predictions must be the average of individual model predictions"
        ))

    def test_base_models_frozen_in_ensemble(self):
        """All weights inside the constituent models should be non-trainable."""
        from src.models import build_ensemble
        m1 = _tiny_binary_model()
        m2 = _tiny_binary_model()
        ensemble = build_ensemble([m1, m2], input_shape=(32, 32, 3))
        # All layers inside m1 and m2 (accessed via ensemble sub-layers) must be frozen
        for sub_model in [m1, m2]:
            assert not sub_model.trainable, (
                "Constituent models must be frozen (model.trainable = False) "
                "inside the ensemble"
            )


# ── Evaluate model ────────────────────────────────────────────────────────────

class TestEvaluateModel:

    def test_evaluate_importable(self):
        try:
            from src.evaluate import evaluate_model
            assert callable(evaluate_model)
        except ImportError as e:
            pytest.fail(f"evaluate_model not importable: {e}")

    def test_returns_dict(self):
        from src.evaluate import evaluate_model
        model = _tiny_binary_model()
        ds, _ = _fake_test_data()
        result = evaluate_model(model, ds)
        assert isinstance(result, dict), "evaluate_model must return a dict"

    def test_required_keys(self):
        from src.evaluate import evaluate_model
        model = _tiny_binary_model()
        ds, _ = _fake_test_data()
        result = evaluate_model(model, ds)
        required = {"accuracy", "precision", "recall", "f1", "auc",
                    "confusion_matrix", "classification_report"}
        missing = required - set(result.keys())
        assert not missing, f"evaluate_model result is missing keys: {missing}"

    def test_accuracy_in_range(self):
        from src.evaluate import evaluate_model
        model = _tiny_binary_model()
        ds, _ = _fake_test_data()
        result = evaluate_model(model, ds)
        assert 0.0 <= result["accuracy"] <= 1.0, (
            f"accuracy must be in [0, 1], got {result['accuracy']}"
        )

    def test_auc_in_range(self):
        from src.evaluate import evaluate_model
        model = _tiny_binary_model()
        ds, _ = _fake_test_data()
        result = evaluate_model(model, ds)
        assert 0.0 <= result["auc"] <= 1.0, (
            f"AUC must be in [0, 1], got {result['auc']}"
        )

    def test_confusion_matrix_shape(self):
        from src.evaluate import evaluate_model
        model = _tiny_binary_model()
        ds, _ = _fake_test_data()
        result = evaluate_model(model, ds)
        cm = np.array(result["confusion_matrix"])
        assert cm.shape == (2, 2), (
            f"confusion_matrix must be 2x2, got shape {cm.shape}"
        )


# ── Visualisation ─────────────────────────────────────────────────────────────

class TestVisualisations:

    def test_plot_confusion_matrix_runs(self):
        from src.evaluate import plot_confusion_matrix
        import matplotlib.pyplot as plt
        cm = np.array([[50, 10], [5, 100]])
        plot_confusion_matrix(cm)
        plt.close("all")

    def test_plot_roc_curve_runs(self):
        from src.evaluate import plot_roc_curve
        import matplotlib.pyplot as plt
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_probs = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])
        plot_roc_curve(y_true, y_probs, model_name="TestModel")
        plt.close("all")

    def test_grad_cam_importable(self):
        try:
            from src.evaluate import grad_cam
            assert callable(grad_cam)
        except ImportError as e:
            pytest.fail(f"grad_cam not importable: {e}")

    def test_grad_cam_output_shape_and_range(self):
        """Grad-CAM must return a 2-D array with values in [0, 1]."""
        from src.evaluate import grad_cam

        # Build a tiny conv model (Grad-CAM requires conv layers)
        inp = keras.Input(shape=(32, 32, 3))
        x = keras.layers.Conv2D(8, 3, activation="relu", name="last_conv")(inp)
        x = keras.layers.GlobalAveragePooling2D()(x)
        out = keras.layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inp, out)

        img = np.random.rand(1, 32, 32, 3).astype("float32")
        heatmap = grad_cam(model, img, last_conv_layer_name="last_conv")

        assert heatmap.ndim == 2, (
            f"Grad-CAM must return a 2-D array, got shape {heatmap.shape}"
        )
        assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0, (
            "Grad-CAM heatmap values must be normalised to [0, 1]"
        )
