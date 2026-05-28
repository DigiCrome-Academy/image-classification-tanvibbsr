"""
Model definitions for all four project phases.

Each function contains TODO stubs that students must implement.
Completed functions return a compiled tf.keras.Model.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


# ════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Baseline MLP  (Sequential API)
# ════════════════════════════════════════════════════════════════════════════

# ── TODO 4 ───────────────────────────────────────────────────────────────────
def build_baseline_mlp(
    input_shape: tuple = (224, 224, 3),
    num_classes: int = 1,
    activation: str = "relu",
    optimizer: str = "adam",
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    Build a baseline Multi-Layer Perceptron using the Sequential API.

    Architecture requirements:
        • Flatten input images.
        • At least 3 Dense layers with decreasing neuron counts
          (e.g., 512 → 256 → 128).
        • Specified hidden-layer activation function.
        • Output layer: 1 neuron + sigmoid (binary classification).
        • Compile with binary_crossentropy loss and accuracy metric.

    Experimentation guidance (do this in your notebook):
        - Try activation='relu', 'leaky_relu', 'elu'.
        - Try optimizer='sgd', 'adam', 'rmsprop', 'nadam'.
        - Add a LearningRateScheduler callback in the training script.

    Args:
        input_shape:   (H, W, C) of input images.
        num_classes:   1 for binary (sigmoid output).
        activation:    Hidden-layer activation name.
        optimizer:     Optimizer name or tf.keras.optimizers instance.
        learning_rate: Initial learning rate.

    Returns:
        Compiled keras.Model
    """
    # ── YOUR CODE STARTS HERE ─────────────────────────────────────────────
    #Flatten the input images
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(512, activation=activation),
        layers.Dense(256, activation=activation),
        layers.Dense(128, activation=activation),
        layers.Dense(num_classes, activation='sigmoid')
    ])
    #Compile the model
    model.compile(
        optimizer=optimizer if isinstance(optimizer, str) else optimizer(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
    raise NotImplementedError("TODO 4: implement build_baseline_mlp()")
    # ── YOUR CODE ENDS HERE ───────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Advanced Neural Network  (Functional API)
# ════════════════════════════════════════════════════════════════════════════

# ── TODO 5 ───────────────────────────────────────────────────────────────────
def build_advanced_nn(
    input_shape: tuple = (224, 224, 3),
    dropout_rate: float = 0.5,
    l2_lambda: float = 1e-4,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    Build a deeper network with regularisation using the Functional API.

    Architecture requirements:
        • Flatten layer after the Input.
        • At least 4 Dense layers with BatchNormalization after each.
        • Dropout layer(s) with dropout_rate.
        • L2 regularisation (kernel_regularizer) on Dense layers.
        • Output: 1 neuron + sigmoid.

    Compile with:
        • Optimizer: Adam with gradient clipping
          (clipnorm=1.0 or clipvalue=0.5).
        • Loss: binary_crossentropy.
        • Metrics: accuracy, AUC.

    Args:
        input_shape:   (H, W, C).
        dropout_rate:  Fraction of neurons to drop.
        l2_lambda:     L2 regularisation coefficient.
        learning_rate: Adam learning rate.

    Returns:
        Compiled keras.Model
    """
    # ── YOUR CODE STARTS HERE ─────────────────────────────────────────────
    inputs = layers.Input(shape=input_shape)
    x = layers.Flatten()(inputs)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC()]
    )
    return model


    # ── YOUR CODE ENDS HERE ───────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Transfer Learning
# ════════════════════════════════════════════════════════════════════════════

# ── TODO 6 ───────────────────────────────────────────────────────────────────
def build_transfer_model(
    base_model_name: str = "MobileNetV2",
    input_shape: tuple = (224, 224, 3),
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-4,
    freeze_base: bool = True,
) -> keras.Model:
    """
    Build a transfer learning model with a frozen (or partially frozen) base.

    Supported base_model_name values: 'VGG16', 'ResNet50', 'MobileNetV2'.

    Architecture requirements:
        1. Load the chosen pretrained base with include_top=False,
           weights='imagenet', and the given input_shape.
        2. If freeze_base=True, set base_model.trainable = False.
        3. Add a custom classification head:
               GlobalAveragePooling2D → Dense(256, relu) →
               BatchNormalization → Dropout(dropout_rate) →
               Dense(1, sigmoid).
        4. Compile with Adam(learning_rate), binary_crossentropy, [accuracy, AUC].

    Fine-tuning guidance (do in notebook after initial training):
        - Unfreeze the top N layers of the base model.
        - Recompile with a lower learning rate (e.g., 1e-5).
        - Continue training for a few more epochs.

    Args:
        base_model_name: One of 'VGG16', 'ResNet50', 'MobileNetV2'.
        input_shape:     (H, W, C) — must match ImageNet preprocessing.
        dropout_rate:    Dropout in the classification head.
        learning_rate:   Initial learning rate.
        freeze_base:     Whether to freeze the base model weights.

    Returns:
        Compiled keras.Model
    """
    # ── YOUR CODE STARTS HERE ─────────────────────────────────────────────
    if base_model_name == "VGG16":
        base_model = keras.applications.VGG16(
            include_top=False, weights='imagenet', input_shape=input_shape)
    elif base_model_name == "ResNet50":
        base_model = keras.applications.ResNet50(
            include_top=False, weights='imagenet', input_shape=input_shape)
    elif base_model_name == "MobileNetV2":
        base_model = keras.applications.MobileNetV2(
            include_top=False, weights='imagenet', input_shape=input_shape)
    else:
        raise ValueError(f"Unsupported base model: {base_model_name}")
    if freeze_base:
        base_model.trainable = False
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=not freeze_base)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC()]
    )
    return model
    # ── YOUR CODE ENDS HERE ───────────────────────────────────────────────


# ── TODO 7 ───────────────────────────────────────────────────────────────────
def unfreeze_top_layers(model: keras.Model, num_layers: int = 20) -> keras.Model:
    """
    Unfreeze the last `num_layers` of the base model for fine-tuning.

    Tasks:
        1. Locate the base model layer inside `model` (it will be the large
           sub-model, identifiable by its layer count).
        2. Freeze all layers except the last `num_layers` of that sub-model.
        3. Recompile the model with Adam(1e-5) and the same loss/metrics.
        4. Return the updated model.

    Args:
        model:      A compiled transfer learning model from build_transfer_model.
        num_layers: Number of top base-model layers to unfreeze.

    Returns:
        Re-compiled keras.Model
    """
    # ── YOUR CODE STARTS HERE ─────────────────────────────────────────────
    #unfreeze the last `num_layers` of the base model for fine-tuning
        # Locate the base model layer inside `model`
    base_model = None
    for layer in model.layers:
        if isinstance(layer, keras.Model) and len(layer.layers) > 50:  # Heuristic for base model
            base_model = layer
            break
    if base_model is None:
        raise ValueError("Base model not found in the provided model.")
    # Freeze all layers except the last `num_layers` of that sub-model
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    for layer in base_model.layers[-num_layers:]:
        layer.trainable = True
    # Recompile the model with Adam(1e-5) and the same loss/metrics
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=model.loss,
        metrics=model.metrics
    )
    return model
  
    # ── YOUR CODE ENDS HERE ───────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Ensemble
# ════════════════════════════════════════════════════════════════════════════

# ── TODO 8 ───────────────────────────────────────────────────────────────────
def build_ensemble(models: list, input_shape: tuple = (224, 224, 3)) -> keras.Model:
    """
    Combine multiple trained models into a soft-voting ensemble.

    Tasks:
        1. Freeze all layers in each model in `models`.
        2. Create a shared Input layer with input_shape.
        3. Pass the input through each model to get its probability output.
        4. Average the outputs using layers.Average().
        5. Return the (uncompiled) ensemble Model — evaluation will be done
           with model.predict() in the notebook.

    Args:
        models:      List of trained keras.Model instances.
        input_shape: (H, W, C).

    Returns:
        keras.Model (not compiled — outputs averaged probability)
    """
    # ── YOUR CODE STARTS HERE ─────────────────────────────────────────────
    # Freeze all layers in each model
    for model in models:
        model.trainable = False
    # Create a shared Input layer
    ensemble_input = layers.Input(shape=input_shape)
    # Pass the input through each model to get its probability output
    model_outputs = [model(ensemble_input) for model in models]
    # Average the outputs
    averaged_output = layers.Average()(model_outputs)
    # Return the ensemble Model
    ensemble_model = keras.Model(inputs=ensemble_input, outputs=averaged_output)
    return ensemble_model
  
    # ── YOUR CODE ENDS HERE ───────────────────────────────────────────────
