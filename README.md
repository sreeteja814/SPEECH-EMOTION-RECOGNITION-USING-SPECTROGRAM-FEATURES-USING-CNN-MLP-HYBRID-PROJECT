# 🎙️ Speech Emotion Recognition — CNN + MLP on RAVDESS

A deep-learning system that classifies **8 emotions** from speech audio using a hybrid **CNN + MLP** architecture trained on the **RAVDESS** dataset, with a polished **Streamlit** interface.

---

## 📁 Project Structure

```
speech_emotion_recognition/
├── feature_extraction.py   # MFCC, Chroma, Mel, ZCR, RMS + spectrogram extraction
├── model.py                # CNN+MLP architecture + training pipeline
├── inference.py            # Load model & predict on new audio
├── app.py                  # Streamlit web interface
├── requirements.txt
└── models/                 # Created after training
    ├── best_model.keras
    ├── scaler.pkl
    └── spec_shape.npy
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the RAVDESS dataset
Download from [Zenodo](https://zenodo.org/record/1188976) — *Audio_Speech_Actors_01-24.zip*

Extract so your folder looks like:
```
data/
└── RAVDESS/
    ├── Actor_01/
    │   ├── 03-01-01-01-01-01-01.wav
    │   └── ...
    ├── Actor_02/
    └── ...
```

### 3. Train the model
```bash
python model.py --data data/RAVDESS --epochs 120 --batch 32 --save models
```

Training takes ~10–20 min on CPU, ~3–5 min on GPU.  
Expect **~70–78% test accuracy** with the default settings.

### 4. Launch the Streamlit app
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧠 Architecture

```
INPUT
 ├── CNN Branch (Log-Mel Spectrogram 64×T×1)
 │   Conv2D(32)→BN→ReLU→MaxPool
 │   Conv2D(64)→BN→ReLU→MaxPool
 │   Conv2D(128)→BN→ReLU→MaxPool
 │   Conv2D(256)→BN→ReLU→GlobalAvgPool
 │   Dense(256)→Dropout(0.4)
 │
 └── MLP Branch (148-d hand-crafted features)
     Dense(256)→BN→ReLU→Dropout(0.3)
     Dense(128)→BN→ReLU→Dropout(0.3)
     Dense(64)→ReLU
           │
     Concatenate (320-d)
     Dense(256)→Dropout(0.4)
     Dense(128)→Dropout(0.3)
     Dense(8, softmax)
```

### Features extracted (148-d)
| Feature | Coefficients | Stats | Total |
|---------|-------------|-------|-------|
| MFCC    | 40          | mean+std | 80 |
| Chroma  | 12          | mean+std | 24 |
| Mel (means only) | 40 | mean | 40 |
| ZCR     | 1           | mean+std | 2  |
| RMS     | 1           | mean+std | 2  |
| **Total** | | | **148** |

---

## 🎭 Emotions Classified

| Code | Emotion | Emoji |
|------|---------|-------|
| 01   | Neutral  | 😐 |
| 02   | Calm     | 😌 |
| 03   | Happy    | 😄 |
| 04   | Sad      | 😢 |
| 05   | Angry    | 😠 |
| 06   | Fearful  | 😨 |
| 07   | Disgust  | 🤢 |
| 08   | Surprised| 😲 |

---

## ⚙️ Training Tips

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--epochs` | 80 | EarlyStopping kicks in ~40-60 |
| `--batch`  | 32 | Reduce to 16 if OOM on GPU |
| `--data`   | data/RAVDESS | Path to RAVDESS root |

- **Class imbalance** is handled via `class_weight="balanced"`
- **LR scheduling**: ReduceLROnPlateau (factor=0.5, patience=7)
- **Regularisation**: BatchNorm + Dropout + EarlyStopping (patience=15)

---

## 📊 Expected Performance (RAVDESS, 8-class)

| Metric | Typical Range |
|--------|-------------|
| Test Accuracy | 70–78% |
| Calm vs Neutral | Hardest pair |
| Angry / Happy | Easiest to classify |

---

## 🔧 Customisation

**Change the number of MFCC coefficients:**
```python
# feature_extraction.py → extract_features()
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # change 40
```

**Use only 4 emotions (angry/happy/sad/neutral):**
```python
# feature_extraction.py → EMOTION_MAP
EMOTION_MAP = {"01": "neutral", "03": "happy", "04": "sad", "05": "angry"}
```

**Adjust CNN depth:**
```python
# model.py → build_cnn_branch()
# Add or remove Conv2D blocks
```