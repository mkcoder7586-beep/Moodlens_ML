import os
import pandas as pd
from datasets import load_dataset

# Get project root (one level above src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw")

# Ensure directory exists
os.makedirs(DATA_PATH, exist_ok=True)

def load_goemotions():
    file_path = os.path.join(DATA_PATH, "goemotions.csv")
    
    if os.path.exists(file_path):
        print("Loading GoEmotions from local...")
        return pd.read_csv(file_path)
    
    print("Downloading GoEmotions...")
    dataset = load_dataset("go_emotions")
    
    df = pd.DataFrame(dataset['train'])
    df.to_csv(file_path, index=False)
    
    return df


def load_hf_emotion():
    file_path = os.path.join(DATA_PATH, "hf_emotion.csv")
    
    if os.path.exists(file_path):
        print("Loading HF Emotion from local...")
        return pd.read_csv(file_path)
    
    print("Downloading HF Emotion dataset...")
    dataset = load_dataset("emotion")
    
    df = pd.DataFrame(dataset['train'])
    df.to_csv(file_path, index=False)
    
    return df