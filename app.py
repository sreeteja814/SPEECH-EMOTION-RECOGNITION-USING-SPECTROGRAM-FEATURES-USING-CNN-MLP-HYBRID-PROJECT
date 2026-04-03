"""
Speech Emotion Recognition — Streamlit App
==========================================
Run:  streamlit run app.py
"""

import os, io, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import soundfile as sf

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title   { font-size:2.4rem; font-weight:700; color:#4F8BF9; }
    .sub-title    { font-size:1.1rem; color:#888; margin-bottom:1.5rem; }
    .emotion-card {
        background: linear-gradient(135deg,#1e3a5f,#2d6a9f);
        border-radius:16px; padding:24px; text-align:center;
        box-shadow:0 4px 20px rgba(0,0,0,0.4);
    }
    .emotion-name { font-size:2.5rem; font-weight:800; color:#fff; }
    .conf-text    { font-size:1.2rem; color:#a0cfff; margin-top:4px; }
    .metric-box   {
        background:#1a1a2e; border-radius:10px; padding:14px;
        text-align:center; border:1px solid #333;
    }
    .stProgress > div > div { background:#4F8BF9; }
</style>
""", unsafe_allow_html=True)

# ── Emotion metadata ──────────────────────────────────────────────────────────
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry",
            "fearful", "disgust", "surprised"]

EMOTION_EMOJI = {
    "neutral":   "😐", "calm":     "😌", "happy":    "😄",
    "sad":       "😢", "angry":    "😠", "fearful":  "😨",
    "disgust":   "🤢", "surprised":"😲",
}

EMOTION_COLOR = {
    "neutral": "#9E9E9E", "calm":     "#64B5F6", "happy":    "#FFD54F",
    "sad":     "#5C6BC0", "angry":    "#EF5350", "fearful":  "#AB47BC",
    "disgust": "#66BB6A", "surprised":"#FF7043",
}

# ── Model loader (cached) ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained CNN+MLP model. Returns None if not yet trained."""
    try:
        from inference import load_model_and_scaler
        model, scaler, n_mels, T = load_model_and_scaler("models")
        return model, scaler, n_mels, T
    except FileNotFoundError:
        return None

# ── Helper: audio feature viz ─────────────────────────────────────────────────
def plot_waveform_and_spectrogram(audio_bytes: bytes):
    """Return a matplotlib figure with waveform + log-Mel spectrogram."""
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, duration=3.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3), facecolor="#0e1117")
    for ax in axes:
        ax.set_facecolor("#0e1117")

    # Waveform
    librosa.display.waveshow(y, sr=sr, ax=axes[0], color="#4F8BF9")
    axes[0].set_title("Waveform", color="white", fontsize=11)
    axes[0].tick_params(colors="white"); axes[0].spines[:].set_color("#333")
    axes[0].set_xlabel("Time (s)", color="#aaa")
    axes[0].set_ylabel("Amplitude", color="#aaa")

    # Log-Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(log_mel, sr=sr, hop_length=512,
                                   x_axis="time", y_axis="mel",
                                   ax=axes[1], cmap="magma")
    axes[1].set_title("Log-Mel Spectrogram", color="white", fontsize=11)
    axes[1].tick_params(colors="white"); axes[1].spines[:].set_color("#333")
    axes[1].set_xlabel("Time (s)", color="#aaa")
    axes[1].set_ylabel("Mel Frequency", color="#aaa")
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB").ax.yaxis.set_tick_params(color="white")

    plt.tight_layout()
    return fig


def plot_probability_bar(probs: np.ndarray):
    """Horizontal bar chart of class probabilities."""
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    colors = [EMOTION_COLOR[e] for e in EMOTIONS]
    y_pos  = range(len(EMOTIONS))
    bars   = ax.barh(list(y_pos), probs * 100, color=colors, height=0.6)

    # Highlight top prediction
    top = int(np.argmax(probs))
    bars[top].set_edgecolor("white"); bars[top].set_linewidth(2)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([f"{EMOTION_EMOJI[e]} {e}" for e in EMOTIONS],
                        color="white", fontsize=10)
    ax.set_xlabel("Probability (%)", color="#aaa", fontsize=10)
    ax.tick_params(axis="x", colors="#aaa")
    ax.spines[:].set_color("#333")
    ax.set_xlim(0, 105)

    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{prob*100:.1f}%", va="center", ha="left",
                color="white", fontsize=9)

    plt.tight_layout()
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/Waveform.svg/320px-Waveform.svg.png",
             use_column_width=True)
    st.markdown("## 🎙️ SER System")
    st.markdown("**CNN + MLP Hybrid Model**")
    st.markdown("---")
    st.markdown("### Detectable Emotions")
    for e in EMOTIONS:
        st.markdown(f"{EMOTION_EMOJI[e]} **{e.capitalize()}**")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
1. **Upload** a `.wav` audio file  
2. Features extracted:  
   - Log-Mel Spectrogram (CNN)  
   - MFCC + Chroma + ZCR + RMS (MLP)  
3. Both branches **fused** → softmax  
4. Emotion **predicted** with confidence
""")
    st.markdown("---")
    st.markdown("### Training")
    st.code("python model.py --data data/RAVDESS", language="bash")
    st.markdown("Dataset: [RAVDESS](https://zenodo.org/record/1188976)")


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🎙️ Speech Emotion Recognition</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">CNN + MLP Hybrid · Trained on RAVDESS · 8 Emotions</p>',
            unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
model_data = load_model()

if model_data is None:
    st.warning("""
### ⚠️ No trained model found

Train the model first by running:
```bash
python model.py --data path/to/RAVDESS --epochs 80
```
Then restart this app. A demo mode (random predictions) is shown below.
""")
    demo_mode = True
else:
    model, scaler, n_mels, T = model_data
    demo_mode = False
    st.success("✅ Model loaded successfully!")

st.markdown("---")

# ── Upload ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📂 Upload Audio")
    uploaded = st.file_uploader(
        "Upload a WAV file (RAVDESS format preferred)",
        type=["wav", "mp3", "ogg", "flac"],
        help="For best results use 16-48 kHz mono WAV files, ~3 s duration"
    )

    if uploaded:
        audio_bytes = uploaded.read()
        st.audio(audio_bytes, format="audio/wav")

        # Show audio info
        try:
            with io.BytesIO(audio_bytes) as buf:
                y_info, sr_info = librosa.load(buf, sr=None, duration=None)
            duration = len(y_info) / sr_info
            st.markdown(f"**Sample rate:** {sr_info} Hz &nbsp;|&nbsp; **Duration:** {duration:.2f} s &nbsp;|&nbsp; **Samples:** {len(y_info):,}")
        except Exception:
            pass

with col2:
    st.markdown("### 🔍 Prediction")
    if uploaded:
        with st.spinner("Extracting features & predicting…"):
            time.sleep(0.3)  # UX pause

            if demo_mode:
                # Demo mode: random prediction
                probs = np.abs(np.random.dirichlet(np.ones(8)))
                pred_idx     = int(np.argmax(probs))
                predicted    = EMOTIONS[pred_idx]
                confidence   = float(probs[pred_idx])
                top3 = sorted(zip(EMOTIONS, probs), key=lambda x: -x[1])[:3]
            else:
                try:
                    from inference import predict_from_bytes
                    predicted, confidence, top3, probs = predict_from_bytes(
                        audio_bytes, model, scaler, n_mels, T
                    )
                except Exception as err:
                    st.error(f"Prediction failed: {err}")
                    st.stop()

        # Prediction card
        st.markdown(f"""
<div class="emotion-card">
  <div style="font-size:3rem">{EMOTION_EMOJI[predicted]}</div>
  <div class="emotion-name">{predicted.upper()}</div>
  <div class="conf-text">Confidence: {confidence*100:.1f}%</div>
</div>
""", unsafe_allow_html=True)

        # Top-3 emotions
        st.markdown("#### Top Predictions")
        for rank, (emo, prob) in enumerate(top3):
            col_a, col_b, col_c = st.columns([1, 5, 1])
            col_a.markdown(f"**#{rank+1}** {EMOTION_EMOJI[emo]}")
            col_b.progress(float(prob))
            col_c.markdown(f"**{prob*100:.1f}%**")

    else:
        st.info("👆 Upload an audio file to see the emotion prediction here.")

# ── Visualisation ─────────────────────────────────────────────────────────────
if uploaded:
    st.markdown("---")
    st.markdown("### 📊 Audio Analysis")

    tab1, tab2 = st.tabs(["Waveform & Spectrogram", "Emotion Probability Distribution"])

    with tab1:
        fig1 = plot_waveform_and_spectrogram(audio_bytes)
        st.pyplot(fig1, use_container_width=True)

    with tab2:
        if "probs" in dir():
            fig2 = plot_probability_bar(probs)
            st.pyplot(fig2, use_container_width=True)

# ── Model Architecture ────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🏗️ Model Architecture Details"):
    st.markdown("""
```
INPUT
 ├── Spectrogram Branch (CNN)          ├── Feature Branch (MLP)
 │   Input: (64 × T × 1)              │   Input: (148,) — MFCC+Chroma+Mel+ZCR+RMS
 │   Conv2D(32) → BN → ReLU → Pool    │   Dense(256) → BN → ReLU → Dropout(0.3)
 │   Conv2D(64) → BN → ReLU → Pool    │   Dense(128) → BN → ReLU → Dropout(0.3)
 │   Conv2D(128) → BN → ReLU → Pool   │   Dense(64) → ReLU
 │   Conv2D(256) → BN → ReLU          │
 │   GlobalAveragePooling2D            │
 │   Dense(256) → Dropout(0.4)        │
 │                                    │
 └──────────────┬─────────────────────┘
                │ Concatenate (256 + 64 = 320)
            Dense(256) → Dropout(0.4)
            Dense(128) → Dropout(0.3)
            Dense(8, softmax)  ← Emotion output
```
**Total parameters:** ~1.8 M  
**Loss:** Sparse Categorical Cross-Entropy  
**Optimizer:** Adam (lr=1e-3, ReduceLROnPlateau)  
**Regularisation:** BatchNorm + Dropout + Class-weight balancing + EarlyStopping
""")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#555;font-size:0.85rem'>"
    "Built with TensorFlow · Librosa · Streamlit &nbsp;|&nbsp; RAVDESS Dataset"
    "</center>",
    unsafe_allow_html=True
)