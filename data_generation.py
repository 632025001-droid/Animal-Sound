# data_generation.py
import os
import numpy as np
import soundfile as sf

from db_utils import create_db, get_session, AudioSample, drop_db # Import drop_db

DATA_DIR = "data"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

SAMPLE_RATE = 22050

def synthesize_tone(freq, duration=1.5, sr=SAMPLE_RATE, harmonics=3, noise_amp=0.02):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    signal = np.zeros_like(t)
    for h in range(1, harmonics+1):
        signal += (1.0/h) * np.sin(2*np.pi*freq*h*t)
    signal /= np.max(np.abs(signal)) + 1e-9
    # add short bursts (to simulate chirp/grunt)
    burst = np.sin(2*np.pi*(freq*1.5)*t) * (np.exp(-5*(t-duration/2)**2))
    signal += 0.3 * burst
    signal += noise_amp * np.random.randn(len(t))
    signal = signal / (np.max(np.abs(signal)) + 1e-9)
    return signal

def generate_dataset(n_per_class=50):
    drop_db() # Drop existing tables
    create_db()  # ensure db exists
    session = get_session()
    classes = {
        "dog": [250, 400],   # typical energy frequencies (simulated)
        "cat": [400, 700],
        "bird": [1000, 3000]
    }
    idx = 0
    for label, freqs in classes.items():
        for i in range(n_per_class):
            base_freq = float(np.random.uniform(freqs[0], freqs[1]))
            harmonics = np.random.randint(1, 5)
            duration = float(np.random.uniform(0.8, 2.5))
            noise_amp = float(np.random.uniform(0.005, 0.05))
            audio = synthesize_tone(base_freq, duration=duration, harmonics=harmonics, noise_amp=noise_amp)
            filename = f"{label}_{i}.wav"
            filepath = os.path.join(AUDIO_DIR, filename)
            sf.write(filepath, audio, SAMPLE_RATE)
            sample = AudioSample(filename=filepath, label=label, duration=duration, sr=SAMPLE_RATE)
            session.add(sample)
            idx += 1
    session.commit()
    session.close()
    print(f"Generated {idx} audio files under {AUDIO_DIR}")

if __name__ == "__main__":
    generate_dataset(n_per_class=60)  # contoh 60 per class = 180 samples
