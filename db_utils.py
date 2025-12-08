# db_utils.py
import os
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class AudioSample(Base):
    __tablename__ = "audio_samples"
    id = Column(Integer, primary_key=True)
    filename = Column(String, unique=True, nullable=False)
    label = Column(String, nullable=False)
    mfcc_path = Column(String, nullable=True)  # path to saved MFCC .npy
    duration = Column(Float, nullable=True)
    sr = Column(Integer, nullable=True)

def get_engine(db_path="sqlite:///db.sqlite"):
    return create_engine(db_path, echo=False, connect_args={"check_same_thread": False})

def create_db(db_path="sqlite:///db.sqlite"):
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return engine

def drop_db(db_path="sqlite:///db.sqlite"):
    engine = get_engine(db_path)
    Base.metadata.drop_all(engine)

def get_session(db_path="sqlite:///db.sqlite"):
    engine = get_engine(db_path)
    Session = sessionmaker(bind=engine)
    return Session()
