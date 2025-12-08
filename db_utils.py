# db_utils.py
import os
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Float, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker

# Determine project base directory (fallback to cwd)
try:
    BASE_DIR = Path(__file__).resolve().parent
except Exception:
    BASE_DIR = Path(os.getcwd())

DB_PATH = BASE_DIR / "db.sqlite"
DB_URL = f"sqlite:///{DB_PATH}"

Base = declarative_base()

class AudioSample(Base):
    __tablename__ = "audio_samples"
    id = Column(Integer, primary_key=True)
    filename = Column(String, unique=True, nullable=False)  # absolute path
    label = Column(String, nullable=False)
    mfcc_path = Column(String, nullable=True)  # absolute path to .npy
    duration = Column(Float, nullable=True)
    sr = Column(Integer, nullable=True)

def get_engine(db_url: str = None):
    url = db_url if db_url else DB_URL
    # disable check_same_thread for multithreaded apps
    return create_engine(url, echo=False, connect_args={"check_same_thread": False})

def create_db(reset: bool = False, db_url: str = None):
    engine = get_engine(db_url)
    # optionally remove existing file
    if reset:
        try:
            path = Path(DB_PATH)
            if path.exists():
                path.unlink()
        except Exception:
            pass
    Base.metadata.create_all(engine)
    return engine

def get_session(db_url: str = None):
    engine = get_engine(db_url)
    Session = sessionmaker(bind=engine)
    return Session()

# helper to add sample safely
def add_sample(session, filename: str, label: str, duration: float = None, sr: int = None):
    filename = str(Path(filename).resolve())
    existing = session.query(AudioSample).filter_by(filename=filename).first()
    if existing:
        return existing
    s = AudioSample(filename=filename, label=label, duration=duration, sr=sr)
    session.add(s)
    session.commit()
    return s
