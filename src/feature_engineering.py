import pandas as pd
from preprocessing import YES_NO_COLS
SYMPTOM_FLAG_COLS = [c + '_flag' for c in YES_NO_COLS]
NUMERIC_COLS = ['age_scaled', 'bp_scaled', 'chol_scaled']

def build_disease_profiles(df: pd.DataFrame) -> pd.DataFrame:
    agg_dict = {col: 'mean' for col in SYMPTOM_FLAG_COLS}
    agg_dict.update({col: 'mean' for col in NUMERIC_COLS})
    profiles = df.groupby('disease').agg(agg_dict)
    profiles['case_count'] = df.groupby('disease').size()
    profiles['typical_risk_level'] = df.groupby('disease')['risk_level'].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else 'Unknown')
    return profiles

def build_patient_feature_vector(symptom_dict: dict) -> pd.DataFrame:
    row = {'fever_flag': symptom_dict.get('fever', 0), 'cough_flag': symptom_dict.get('cough', 0), 'fatigue_flag': symptom_dict.get('fatigue', 0), 'difficulty_breathing_flag': symptom_dict.get('difficulty_breathing', 0), 'age_scaled': symptom_dict.get('age', 30), 'bp_scaled': symptom_dict.get('blood_pressure', 1), 'chol_scaled': symptom_dict.get('cholesterol_level', 1)}
    return pd.DataFrame([row])
if __name__ == '__main__':
    from preprocessing import load_raw_data, clean_data
    raw = load_raw_data('../data/Cleaned_Dataset.csv')
    cleaned = clean_data(raw)
    profiles = build_disease_profiles(cleaned)
    print(profiles.head())
