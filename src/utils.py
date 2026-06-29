import os
import pandas as pd
from preprocessing import load_raw_data, clean_data
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'Cleaned_Dataset.csv')

def get_clean_dataset(path: str=DATA_PATH) -> pd.DataFrame:
    raw = load_raw_data(path)
    return clean_data(raw)

def format_disease_name(name: str) -> str:
    return str(name).strip().title()
