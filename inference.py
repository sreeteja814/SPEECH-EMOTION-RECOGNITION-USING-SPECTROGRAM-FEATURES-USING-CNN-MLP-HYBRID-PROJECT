"""
Inference utilities — load a trained CNN+MLP model and predict on new audio.
No changes needed here; feature dim is auto-detected from saved scaler/model.
"""

import numpy as np
import pickle, os
import tensorflow as tf

from feature_extraction import (extract_features, extract_spectrogram,
                                 EMOTIONS, EMOTION_MAP)


def load_model_and_scaler(model_dir: str = "models"):
    """Load saved Keras model + scaler + spec time dimension."""
    model_path  = os.path.join(model_dir, "best_model.keras")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    shape_path  = os.path.join(model_dir, "spec_shape.npy")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at '{model_path}'.\n"
            "Run `python model.py --data <path/to/RAVDESS>` first."
        )

    model = tf.keras.models.load_model(model_path)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    spec_shape = tuple(np.load(shape_path))   # (n_mels, T, 1)
    n_mels = spec_shape[0]
    T      = spec_shape[1]

    return model, scaler, n_mels, T


def predict_emotion(file_path: str, model, scaler,
                    n_mels: int = 64, T: int = 130, top_k: int = 3):
    """
    Predict emotion for a single audio file.

    Returns:
        predicted_emotion : str
        confidence        : float  (0–1)
        top_k_emotions    : list[(emotion, prob)]
        probs             : np.ndarray  (8,)
    """
    feat = extract_features(file_path)
    spec = extract_spectrogram(file_path, n_mels=n_mels)

    if feat is None or spec is None:
        raise ValueError("Could not extract features from the audio file.")

    # Pad / crop spectrogram to training time dimension T
    if spec.shape[1] < T:
        spec = np.pad(spec, ((0, 0), (0, T - spec.shape[1]), (0, 0)))
    else:
        spec = spec[:, :T, :]

    feat_scaled = scaler.transform(feat.reshape(1, -1))

    probs = model.predict(
        [spec[np.newaxis, ...], feat_scaled], verbose=0
    )[0]

    pred_idx   = int(np.argmax(probs))
    predicted  = EMOTIONS[pred_idx]
    confidence = float(probs[pred_idx])

    top_k_indices  = np.argsort(probs)[::-1][:top_k]
    top_k_emotions = [(EMOTIONS[i], float(probs[i])) for i in top_k_indices]

    return predicted, confidence, top_k_emotions, probs


def predict_from_bytes(audio_bytes: bytes, model, scaler,
                       n_mels: int = 64, T: int = 130):
    """
    Predict emotion directly from raw audio bytes (wav).
    Saves to a temp file then calls predict_emotion.
    """
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        result = predict_emotion(tmp_path, model, scaler, n_mels, T)
    finally:
        os.remove(tmp_path)
    return result