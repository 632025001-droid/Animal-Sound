# app_streamlit.py
import os
import sys
from pathlib import Path
import joblib
import librosa
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from db_utils import get_session, AudioSample, BASE_DIR

# Ensure project base dir in path (helps when running from other working dirs)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

MODEL_PATH = BASE_DIR / "models" / "rf_model.joblib"
FEATURE_DIR = BASE_DIR / "data" / "mfcc"

st.set_page_config(page_title="Mini AI - Animal Sound Classifier", layout="wide")

@st.cache_data
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(str(MODEL_PATH))
    return None

@st.cache_data
def list_samples():
    session = get_session()
    items = session.query(AudioSample).all()
    session.close()
    return items

def resolve_path(path_str):
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    # try relative to BASE_DIR
    candidate = BASE_DIR / path_str
    if candidate.exists():
        return str(candidate.resolve())
    # as last resort, return as-is (librosa will fail if truly missing)
    return str(p)

def extract_feat_for_infer(y, sr, n_mfcc=20):
    y, _ = librosa.effects.trim(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std]), mfcc

model = load_model()

st.title("Mini AI — Klasifikasi Hewan dari Suara")

col1, col2 = st.columns([2,1])

with col1:
    st.header("Input")
    samples = list_samples()
    sample_map = {os.path.basename(s.filename): s for s in samples}
    choice = st.selectbox("Pilih contoh audio dari dataset (atau pilih Upload)", ["-- Upload --"] + sorted(sample_map.keys()))
    uploaded = st.file_uploader("Atau upload file .wav/.mp3 untuk diuji", type=["wav","mp3","flac"])
    input_audio = None
    sr = 22050

    if uploaded is not None:
        # streamlit's UploadedFile provides getbuffer
        import io, soundfile as sf
        data_bytes = uploaded.getbuffer()
        try:
            data, sr = sf.read(io.BytesIO(data_bytes))
            if data.ndim > 1:
                data = data.mean(axis=1)
            input_audio = (data, sr)
        except Exception as e:
            st.error(f"Failed to read uploaded audio: {e}")
    elif choice != "-- Upload --":
        sel = sample_map[choice]
        resolved = resolve_path(sel.filename)
        if not Path(resolved).exists():
            st.error(f"Sample file not found: {resolved}")
        else:
            try:
                y, sr = librosa.load(resolved, sr=sel.sr or 22050)
                input_audio = (y, sr)
            except Exception as e:
                st.error(f"Failed to load sample: {e}")

    if st.button("Predict") and input_audio is not None:
        y, sr = input_audio
        feat_vec, mfcc_matrix = extract_feat_for_infer(y, sr)
        if model is None:
            st.error("Model belum ada. Jalankan train_model.py terlebih dahulu.")
        else:
            pred = model.predict([feat_vec])[0]
            proba = model.predict_proba([feat_vec])[0]
            classes = model.classes_
            st.success(f"Prediksi: **{pred}**")
            df_probs = {c: float(p) for c,p in zip(classes, proba)}
            st.json(df_probs)
            # audio playback
            import soundfile as sf, io
            buf = io.BytesIO()
            sf.write(buf, y, sr, format='WAV')
            st.audio(buf.getvalue(), format="audio/wav")
            # plots
            fig, axs = plt.subplots(2,1, figsize=(8,6))
            axs[0].plot(y); axs[0].set_title("Waveform")
            im = axs[1].imshow(mfcc_matrix, origin='lower', aspect='auto')
            axs[1].set_title("MFCC")
            fig.colorbar(im, ax=axs[1])
            st.pyplot(fig)

with col2:
    st.header("Dataset & Model Info")
    if MODEL_PATH.exists():
        st.write("Model ter-load:", str(MODEL_PATH))
    else:
        st.warning("Model belum dilatih. Jalankan `python train_model.py`.")

    samples = list_samples()
    st.write("Jumlah sampel di DB:", len(samples))
    if samples:
        import pandas as pd
        df = pd.Series([s.label for s in samples]).value_counts().rename_axis('label').reset_index(name='count')
        st.bar_chart(df.set_index('label'))

    st.markdown("---")
    st.write("Instruksi singkat:")
    st.write("""
    1. Jika belum ada data: jalankan `python data_generation.py` (opsional: tambahkan argumen untuk jumlah).
    2. Jalankan `python feature_extraction.py`.
    3. Jalankan `python train_model.py`.
    4. Jalankan `streamlit run app_streamlit.py`.
    """)

st.sidebar.header("Utilities")
if st.sidebar.button("Re-scan DB and refresh"):
    st.experimental_rerun()
