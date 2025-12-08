# app_streamlit.py
import streamlit as st
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import joblib
from db_utils import get_session, AudioSample

MODEL_PATH = "models/rf_model.joblib"
FEATURE_DIR = "data/mfcc"

st.set_page_config(page_title="Mini AI - Animal Sound Classifier", layout="wide")

@st.cache_data
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

@st.cache_data
def list_samples():
    session = get_session()
    items = session.query(AudioSample).all()
    session.close()
    return items

def extract_feat_for_infer(y, sr, n_mfcc=13):
    y, _ = librosa.effects.trim(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std]), mfcc

model = load_model()
st.title("Mini AI — Klasifikasi Hewan dari Suara")
col1, col2 = st.columns(2)

with col1:
    st.header("Input")
    samples = list_samples()
    sample_map = {os.path.basename(s.filename): s for s in samples}
    choice = st.selectbox("Pilih contoh audio dari dataset (atau pilih Upload)", ["-- Upload --"] + list(sample_map.keys()))
    uploaded = st.file_uploader("Atau upload file .wav (mono) untuk diuji", type=["wav","mp3","flac"])
    input_audio = None
    sr = 22050
    if uploaded is not None:
        import io, soundfile as sf
        data, sr = sf.read(io.BytesIO(uploaded.read()))
        if data.ndim > 1:
            data = data.mean(axis=1)
        input_audio = (data, sr)
    elif choice != "-- Upload --":
        sel = sample_map[choice]
        y, sr = librosa.load(sel.filename, sr=sel.sr or 22050)
        input_audio = (y, sr)

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
            # show waveform and MFCC
            st.audio(librosa.util.buf_to_float(y), format='audio/wav')
            fig, ax = plt.subplots(2,1, figsize=(8,6))
            ax[0].plot(y); ax[0].set(title="Waveform")
            im = ax[1].imshow(mfcc_matrix, origin='lower', aspect='auto')
            ax[1].set(title="MFCC")
            fig.colorbar(im, ax=ax[1])
            st.pyplot(fig)

with col2:
    st.header("Dataset & Model Info")
    if os.path.exists(MODEL_PATH):
        st.write("Model ter-load:", MODEL_PATH)
    else:
        st.warning("Model belum dilatih. Jalankan `train_model.py`.")
    # show class distribution
    items = samples
    if items:
        labels = [s.label for s in items]
        import pandas as pd
        df = pd.Series(labels).value_counts().rename_axis('label').reset_index(name='count')
        st.bar_chart(df.set_index('label'))
    st.write("Jumlah sampel di DB:", len(items))

    st.markdown("---")
    st.write("Petunjuk:")
    st.write("""
    1. Jika belum ada data, jalankan `python data_generation.py` untuk membuat audio dummy.  
    2. Jalankan `python feature_extraction.py` untuk menghasilkan MFCC dan menyimpannya.  
    3. Jalankan `python train_model.py` untuk melatih model dan menyimpannya.  
    4. Jalankan `streamlit run app_streamlit.py` untuk membuka dashboard.
    """)

st.sidebar.header("Utilities")
if st.sidebar.button("Re-scan DB and show samples"):
    st.experimental_rerun()
