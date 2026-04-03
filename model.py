"""
CNN + MLP Hybrid Model for Speech Emotion Recognition
------------------------------------------------------
Updates vs original:
  - Deeper CNN (double conv blocks for 32 & 64 filters)
  - BN after final Dense in CNN branch
  - AdamW optimizer + label smoothing (0.1)
  - Stronger EarlyStopping (patience 20) & ReduceLROnPlateau (factor 0.3)
  - Default epochs raised to 120

Architecture:
  ┌─────────────────────┐   ┌──────────────────┐
  │  Spectrogram Input  │   │  Feature Input   │
  │  (64 × T × 1)       │   │  (188,)          │
  └────────┬────────────┘   └────────┬─────────┘
           │ CNN branch               │ MLP branch
    Conv2D(32)×2→BN→ReLU→Pool  Dense(256)→BN→ReLU→Drop(0.3)
    Conv2D(64)×2→BN→ReLU→Pool  Dense(128)→BN→ReLU→Drop(0.3)
    Conv2D(128)→BN→ReLU→Pool   Dense(64)→ReLU
    Conv2D(256)→BN→ReLU
    GlobalAvgPool
    Dense(256)→BN→Drop(0.4)
           │                         │
           └──────────┬──────────────┘
                  Concatenate
               Dense(256)→Drop(0.4)
               Dense(128)→Drop(0.3)
               Dense(8, softmax)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle, os

from feature_extraction import EMOTIONS, NUM_CLASSES


# ── Model definition ──────────────────────────────────────────────────────────

def build_cnn_branch(input_shape):
    """Deeper CNN branch processing log-Mel spectrogram."""
    inp = layers.Input(shape=input_shape, name="spec_input")

    # Block 1 — double conv (32)
    x = layers.Conv2D(32, (3, 3), padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # Block 2 — double conv (64)
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 3 — single conv (128)
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # Block 4 — single conv (256) + global pool
    x = layers.Conv2D(256, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Head
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)   # ← added BN
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)

    return inp, x


def build_mlp_branch(input_dim):
    """MLP branch processing hand-crafted features."""
    inp = layers.Input(shape=(input_dim,), name="feat_input")

    x = layers.Dense(256)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation="relu")(x)

    return inp, x


def build_cnn_mlp_model(spec_shape, feat_dim, num_classes=NUM_CLASSES, lr=1e-3):
    """Combine CNN and MLP branches into a unified model."""
    cnn_inp, cnn_out = build_cnn_branch(spec_shape)
    mlp_inp, mlp_out = build_mlp_branch(feat_dim)

    # Fusion
    merged = layers.Concatenate(name="fusion")([cnn_out, mlp_out])
    x = layers.Dense(256, activation="relu")(merged)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation="softmax", name="emotion_output")(x)

    model = Model(inputs=[cnn_inp, mlp_inp], outputs=output, name="CNN_MLP_SER")

    model.compile(
        # AdamW: Adam + weight-decay regularisation
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4),
        # Sparse labels are integer-encoded; use sparse cross-entropy directly
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model


# ── Training pipeline ─────────────────────────────────────────────────────────

def train(X_spec, X_feat, y, save_dir="models", epochs=120, batch_size=32):
    os.makedirs(save_dir, exist_ok=True)

    # Scale hand-crafted features
    scaler = StandardScaler()
    X_feat_scaled = scaler.fit_transform(X_feat)

    # Train / val / test split  (70 / 15 / 15)
    X_s_tr, X_s_tmp, X_f_tr, X_f_tmp, y_tr, y_tmp = train_test_split(
        X_spec, X_feat_scaled, y,
        test_size=0.30, random_state=42, stratify=y)
    X_s_val, X_s_te, X_f_val, X_f_te, y_val, y_te = train_test_split(
        X_s_tmp, X_f_tmp, y_tmp,
        test_size=0.50, random_state=42, stratify=y_tmp)

    spec_shape = X_s_tr.shape[1:]   # (64, T, 1)
    feat_dim   = X_f_tr.shape[1]    # 188

    print(f"Train: {len(y_tr)}  |  Val: {len(y_val)}  |  Test: {len(y_te)}")
    print(f"Spec shape: {spec_shape}  |  Feat dim: {feat_dim}")

    model = build_cnn_mlp_model(spec_shape, feat_dim)
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=20,                 # was 15
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,                  # was 0.5 — more aggressive drop
            patience=8,                  # was 7
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(save_dir, "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
    ]

    history = model.fit(
        [X_s_tr, X_f_tr], y_tr,
        validation_data=([X_s_val, X_f_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=_class_weights(y_tr),
    )

    # Evaluate on held-out test set
    loss, acc = model.evaluate([X_s_te, X_f_te], y_te, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}  |  Test loss: {loss:.4f}")

    # Save artefacts
    with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    model.save(os.path.join(save_dir, "best_model.keras"))
    np.save(os.path.join(save_dir, "spec_shape.npy"), np.array(spec_shape))

    print(f"Model and scaler saved to '{save_dir}/'")
    return model, history, scaler, (X_s_te, X_f_te, y_te)


def _class_weights(y):
    """Inverse-frequency class weights to handle class imbalance."""
    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight("balanced", classes=np.unique(y), y=y)
    return {i: cw[i] for i in range(len(cw))}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from feature_extraction import load_ravdess_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default="data/RAVDESS",
                        help="Path to RAVDESS root folder")
    parser.add_argument("--epochs",  type=int, default=120)
    parser.add_argument("--batch",   type=int, default=32)
    parser.add_argument("--save",    default="models")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable data augmentation")
    args = parser.parse_args()

    X_spec, X_feat, y = load_ravdess_dataset(
        args.data, augment=not args.no_augment
    )
    train(X_spec, X_feat, y,
          save_dir=args.save,
          epochs=args.epochs,
          batch_size=args.batch)