# feature_extraction.py
import os
import numpy as np
import librosa
from db_utils import get_session, AudioSample

DATA_DIR = "data"
FEATURE_DIR = os.path.join(DATA_DIR, "mfcc")
os.makedirs(FEATURE_DIR, exist_ok=True)

def extract_mfcc(filepath, sr=22050, n_mfcc=13, hop_length=512, n_fft=2048):
    y, sr = librosa.load(filepath, sr=sr)
    # trim silence a bit
    y, _ = librosa.effects.trim(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    # take statistics across time frames (mean + std)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    feat = np.concatenate([mfcc_mean, mfcc_std])
    return feat, mfcc

def process_all():
    session = get_session()
    samples = session.query(AudioSample).all()
    print(f"Found {len(samples)} samples in DB")
    for s in samples:
        try:
            feat_vec, mfcc_matrix = extract_mfcc(s.filename, sr=s.sr or 22050)
            base = os.path.splitext(os.path.basename(s.filename))[0]
            vec_path = os.path.join(FEATURE_DIR, base + "_vec.npy")
            mat_path = os.path.join(FEATURE_DIR, base + "_mfcc.npy")
            np.save(vec_path, feat_vec)
            np.save(mat_path, mfcc_matrix)
            s.mfcc_path = vec_path
            session.add(s)
        except Exception as e:
            print(f"Failed to process {s.filename}: {e}")
    session.commit()
    session.close()
    print("Feature extraction done.")

if __name__ == "__main__":
    process_all()
