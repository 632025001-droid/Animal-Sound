# feature_extraction.py
import os
import numpy as np
import librosa
from pathlib import Path
from db_utils import get_session, AudioSample, BASE_DIR

DATA_DIR = BASE_DIR / "data"
FEATURE_DIR = DATA_DIR / "mfcc"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

def extract_mfcc_vector(filepath: str, sr: int = 22050, n_mfcc: int = 20):
    y, sr_loaded = librosa.load(filepath, sr=sr)
    y, _ = librosa.effects.trim(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr_loaded, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    feat = np.concatenate([mfcc_mean, mfcc_std])
    return feat, mfcc

def process_all():
    session = get_session()
    samples = session.query(AudioSample).all()
    print(f"[INFO] Found {len(samples)} samples in DB")
    processed = 0
    for s in samples:
        try:
            # ensure absolute path
            filepath = str(Path(s.filename).resolve())
            if not Path(filepath).exists():
                print(f"[WARN] File not found, skipping: {filepath}")
                continue
            feat_vec, mfcc_mat = extract_mfcc_vector(filepath, sr=s.sr or 22050)
            base = Path(filepath).stem
            vec_path = FEATURE_DIR / (base + "_vec.npy")
            mat_path = FEATURE_DIR / (base + "_mfcc.npy")
            np.save(str(vec_path), feat_vec)
            np.save(str(mat_path), mfcc_mat)
            # save absolute path to DB
            s.mfcc_path = str(vec_path.resolve())
            session.add(s)
            processed += 1
            print(f"[OK] Processed: {filepath}")
        except Exception as e:
            print(f"[ERROR] Failed to process {s.filename}: {e}")
    session.commit()
    session.close()
    print(f"[DONE] Feature extraction finished. Processed: {processed}")

if __name__ == "__main__":
    process_all()
