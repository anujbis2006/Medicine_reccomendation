import pandas as pd
YES_NO_COLS = ['fever', 'cough', 'fatigue', 'difficulty_breathing']

def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f'Removed {before - after} duplicate rows')
    df['disease'] = df['disease'].astype(str).str.strip()
    df['gender'] = df['gender'].astype(str).str.strip().str.lower()
    for col in YES_NO_COLS:
        df[col + '_flag'] = df[col].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
    df = df.reset_index(drop=True)
    return df

def get_missing_value_report(df: pd.DataFrame) -> pd.Series:
    return df.isnull().sum()
if __name__ == '__main__':
    raw = load_raw_data('../data/Cleaned_Dataset.csv')
    cleaned = clean_data(raw)
    print(cleaned.head())
    print(get_missing_value_report(cleaned))
