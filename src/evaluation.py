import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from feature_engineering import SYMPTOM_FLAG_COLS, NUMERIC_COLS
FEATURE_COLS = SYMPTOM_FLAG_COLS + NUMERIC_COLS

def build_small_model(input_dim, num_classes):
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    return model

def cross_validated_accuracy(df, n_splits=3):
    X = df[FEATURE_COLS].values.astype('float32')
    y = LabelEncoder().fit_transform(df['disease'])
    counts = pd.Series(y).value_counts()
    valid_classes = counts[counts >= n_splits].index
    mask = pd.Series(y).isin(valid_classes).values
    X_valid, y_valid = (X[mask], y[mask])
    if len(np.unique(y_valid)) < 2:
        return {'mean_accuracy': None, 'note': 'Not enough data per class to cross-validate.'}
    y_valid_relabel = LabelEncoder().fit_transform(y_valid)
    num_classes = len(np.unique(y_valid_relabel))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in skf.split(X_valid, y_valid_relabel):
        X_train, X_test = (X_valid[train_idx], X_valid[test_idx])
        y_train, y_test = (y_valid_relabel[train_idx], y_valid_relabel[test_idx])
        model = build_small_model(X_train.shape[1], num_classes)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        scores.append(acc)
    scores = np.array(scores)
    return {'mean_accuracy': round(float(scores.mean()), 3), 'std_accuracy': round(float(scores.std()), 3), 'fold_scores': [round(float(s), 3) for s in scores], 'classes_used': len(valid_classes), 'classes_dropped_too_rare': int(df['disease'].nunique() - len(valid_classes))}

def top_k_accuracy(recommender, df, k=3):
    from feature_engineering import build_patient_feature_vector
    hits = 0
    for _, row in df.iterrows():
        symptom_dict = {'fever': row.get('fever_flag', 0), 'cough': row.get('cough_flag', 0), 'fatigue': row.get('fatigue_flag', 0), 'difficulty_breathing': row.get('difficulty_breathing_flag', 0), 'age': row['age_scaled'], 'blood_pressure': row['bp_scaled'], 'cholesterol_level': row['chol_scaled']}
        vector = build_patient_feature_vector(symptom_dict)
        matches = recommender.recommend_for_patient(vector, n=k)
        if row['disease'] in matches['disease'].values:
            hits += 1
    return round(hits / len(df), 3)
if __name__ == '__main__':
    from preprocessing import load_raw_data, clean_data
    from recommender import MedicineRecommender
    import os
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Cleaned_Dataset.csv')
    raw = load_raw_data(data_path)
    cleaned = clean_data(raw)
    print('Cross-validated model accuracy:')
    print(cross_validated_accuracy(cleaned))
    engine = MedicineRecommender(cleaned)
    print('Top-3 content-based match accuracy:')
    print(top_k_accuracy(engine, cleaned, k=3))
