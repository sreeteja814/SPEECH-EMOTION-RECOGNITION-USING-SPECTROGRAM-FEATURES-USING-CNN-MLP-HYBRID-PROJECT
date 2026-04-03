"""
Feature Extraction for RAVDESS Speech Emotion Recognition
Extracts MFCC, Chroma, Mel Spectrogram, ZCR, and RMSE features
— Updated: data augmentation + Mel std bug fix (188-d features)
"""

import numpy as np
import librosa
import os
import glob
from pathlib import Path

# ── RAVDESS label mapping ──────────────────────────────────────────────────────
# Filename format: 03-01-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav
# Emotion codes: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

EMOTIONS    = list(EMOTION_MAP.values())
NUM_CLASSES = len(EMOTIONS)

SR       = 22050
DURATION = 3.0


# ── Augmentation ───────────────────────────────────────────────────────────────

def augment_audio(y: np.ndarray, sr: int = SR):
    """
    Return a list of augmented audio arrays (always includes the original).
    5 versions per file → ~5× training data.
    """
    augmented = [y]   # 1. original

    # 2. Time-stretch slow
    try:
        augmented.append(librosa.effects.time_stretch(y, rate=0.85))
    except Exception:
        pass

    # 3. Time-stretch fast
    try:
        augmented.append(librosa.effects.time_stretch(y, rate=1.15))
    except Exception:
        pass

    # 4. Pitch shift down
    try:
        augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-2))
    except Exception:
        pass

    # 5. Pitch shift up
    try:
        augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=2))
    except Exception:
        pass

    # 6. White noise
    noise = np.random.randn(len(y)).astype(np.float32) * 0.005
    augmented.append(y + noise)

    return augmented   # up to 6 versions


# ── Internal helpers ───────────────────────────────────────────────────────────

def _pad_or_trim(y: np.ndarray, sr: int = SR, duration: float = DURATION) -> np.ndarray:
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y


def _extract_features_from_array(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Extract 188-d feature vector from a numpy audio array.

    Features:
      MFCC (40)      → mean + std = 80
      Chroma (12)    → mean + std = 24
      Mel (40)       → mean + std = 80   ← std was missing before (bug fix)
      ZCR (1)        → mean + std = 2
      RMS (1)        → mean + std = 2
      ─────────────────────────────────
      Total                        188
    """
    try:
        features = []

        # 1. MFCC (80)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc,  axis=1))

        # 2. Chroma STFT (24)
        stft   = np.abs(librosa.stft(y))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma,  axis=1))

        # 3. Mel Spectrogram — mean + std (80)  ← BUG FIX: std was missing
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
        features.extend(np.mean(mel, axis=1))
        features.extend(np.std(mel,  axis=1))   # ← was absent before

        # 4. Zero Crossing Rate (2)
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(float(np.mean(zcr)))
        features.append(float(np.std(zcr)))

        # 5. RMS Energy (2)
        rms = librosa.feature.rms(y=y)
        features.append(float(np.mean(rms)))
        features.append(float(np.std(rms)))

        return np.array(features, dtype=np.float32)   # 188-d

    except Exception as e:
        print(f"[ERROR] _extract_features_from_array: {e}")
        return None


def _extract_spectrogram_from_array(y: np.ndarray, sr: int = SR,
                                     n_mels: int = 64,
                                     hop_length: int = 512) -> np.ndarray:
    """Extract normalised log-Mel spectrogram from a numpy array → (n_mels, T, 1)."""
    try:
        mel     = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                                  hop_length=hop_length)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6)
        return log_mel[..., np.newaxis]   # (n_mels, T, 1)
    except Exception as e:
        print(f"[ERROR] _extract_spectrogram_from_array: {e}")
        return None


# ── Public API (used by inference.py) ─────────────────────────────────────────

def extract_features(file_path: str, sr: int = SR,
                     duration: float = DURATION) -> np.ndarray:
    """Extract 188-d feature vector from an audio file path."""
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=duration, mono=True)
        y = _pad_or_trim(y, sr, duration)
        return _extract_features_from_array(y, sr)
    except Exception as e:
        print(f"[ERROR] extract_features({file_path}): {e}")
        return None


def extract_spectrogram(file_path: str, sr: int = SR, duration: float = DURATION,
                        n_mels: int = 64, hop_length: int = 512) -> np.ndarray:
    """Extract log-Mel spectrogram from an audio file path → (n_mels, T, 1)."""
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=duration, mono=True)
        y = _pad_or_trim(y, sr, duration)
        return _extract_spectrogram_from_array(y, sr, n_mels, hop_length)
    except Exception as e:
        print(f"[ERROR] extract_spectrogram({file_path}): {e}")
        return None


# ── Dataset loader ─────────────────────────────────────────────────────────────

def load_ravdess_dataset(dataset_path: str, n_mels: int = 64,
                          hop_length: int = 512, augment: bool = True):
    """
    Walk the RAVDESS directory tree and return:
      X_spec : (N, n_mels, T, 1)  — spectrograms for CNN
      X_feat : (N, 188)           — hand-crafted features for MLP branch
      y      : (N,)               — integer class labels

    With augment=True each file produces up to 6 versions → ~6× more data.
    """
    wav_files = glob.glob(os.path.join(dataset_path, "**", "*.wav"), recursive=True)
    print(f"Found {len(wav_files)} wav files in '{dataset_path}'")

    specs, feats, labels = [], [], []

    for fp in sorted(wav_files):
        fname = Path(fp).stem
        parts = fname.split("-")
        if len(parts) < 7:
            continue

        emotion_code = parts[2]
        if emotion_code not in EMOTION_MAP:
            continue

        label = list(EMOTION_MAP.keys()).index(emotion_code)

        # Load raw audio once
        try:
            y_raw, sr_raw = librosa.load(fp, sr=SR, duration=DURATION, mono=True)
            y_raw = _pad_or_trim(y_raw, SR, DURATION)
        except Exception as e:
            print(f"[ERROR] loading {fp}: {e}")
            continue

        # Produce original + augmented versions
        versions = augment_audio(y_raw, SR) if augment else [y_raw]

        for aug_y in versions:
            aug_y = _pad_or_trim(aug_y, SR, DURATION)   # re-trim after stretch

            spec = _extract_spectrogram_from_array(aug_y, SR, n_mels, hop_length)
            feat = _extract_features_from_array(aug_y, SR)

            if spec is None or feat is None:
                continue

            specs.append(spec)
            feats.append(feat)
            labels.append(label)

    print(f"Total samples after augmentation: {len(labels)}")

    # ── Uniform time dimension: pad / crop to median T ────────────────────────
    time_frames = [s.shape[1] for s in specs]
    T = int(np.median(time_frames))
    padded = []
    for s in specs:
        if s.shape[1] < T:
            s = np.pad(s, ((0, 0), (0, T - s.shape[1]), (0, 0)))
        else:
            s = s[:, :T, :]
        padded.append(s)

    X_spec = np.array(padded,  dtype=np.float32)
    X_feat = np.array(feats,   dtype=np.float32)
    y      = np.array(labels,  dtype=np.int32)

    print(f"Dataset ready — spec: {X_spec.shape}, feat: {X_feat.shape}, labels: {y.shape}")
    return X_spec, X_feat, y