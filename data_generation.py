# data_generation.py
import os
import numpy as np
import soundfile as sf
from pathlib import Path
from db_utils import create_db, get_session, add_sample, BASE_DIR

# Config
DATA_DIR = BASE_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"
SAMPLE_RATE = 22050

AUDIO_DIR.mkdir(parents=True, exist_ok=True)

def synthesize_tone(freq, duration=1.5, sr=SAMPLE_RATE, harmonics=3, noise_amp=0.02):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    signal = np.zeros_like(t)
    for h in range(1, harmonics+1):
        signal += (1.0/h) * np.sin(2*np.pi*freq*h*t)
    signal /= (np.max(np.abs(signal)) + 1e-9)
    burst = np.sin(2*np.pi*(freq*1.5)*t) * (np.exp(-5*(t-duration/2)**2))
    signal += 0.3 * burst
    signal += noise_amp * np.random.randn(len(t))
    signal = signal / (np.max(np.abs(signal)) + 1e-9)
    return signal.astype(np.float32)

def generate_dataset(n_per_class=50, reset_db=False):
    # ensure DB and session
    create_db(reset=reset_db)
    session = get_session()

    classes = {
        "dog": [250, 400],
        "cat": [400, 700],
        "bird": [1000, 3000]
    }

    total = 0
    for label, freqs in classes.items():
        for i in range(n_per_class):
            base_freq = float(np.random.uniform(freqs[0], freqs[1]))
            harmonics = int(np.random.randint(1, 5))
            duration = float(np.random.uniform(0.8, 2.5))
            noise_amp = float(np.random.uniform(0.005, 0.05))
            audio = synthesize_tone(base_freq, duration=duration, harmonics=harmonics, noise_amp=noise_amp)
            filename = AUDIO_DIR / f"{label}_{i:03d}.wav"
            sf.write(str(filename), audio, SAMPLE_RATE)
            # add to DB (stores absolute path)
            add_sample(session, filename=str(filename), label=label, duration=duration, sr=SAMPLE_RATE)
            total += 1

    session.close()
    print(f"[OK] Generated {total} audio files in: {AUDIO_DIR}")

if __name__ == "__main__":
    # generate 60 per class by default
    generate_dataset(n_per_class=60, reset_db=False)
