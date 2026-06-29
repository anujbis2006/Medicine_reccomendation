import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from medicine_db import MEDICINE_DB, DEFAULT_TREATMENT
from feature_engineering import SYMPTOM_FLAG_COLS, NUMERIC_COLS, build_disease_profiles
FEATURE_COLS = SYMPTOM_FLAG_COLS + NUMERIC_COLS

class MedicineRecommender:

    def __init__(self, df):
        self.df = df
        self.disease_profiles = build_disease_profiles(df)
        self.popularity = df['disease'].value_counts()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.num_classes = df['disease'].nunique()
        self.train_model()

    def train_model(self):
        X = self.df[FEATURE_COLS].values.astype('float32')
        y = self.label_encoder.fit_transform(self.df['disease'])
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(X, y)

    def recommend_top_n_popular(self, n=10):
        return self.popularity.head(n)

    def recommend_similar_diseases(self, disease_name, n=5):
        if disease_name not in self.disease_profiles.index:
            fallback = self.recommend_top_n_popular(n)
            return pd.DataFrame({'disease': fallback.index, 'similarity': np.nan})
        profile_matrix = self.disease_profiles[FEATURE_COLS].values
        sim_matrix = cosine_similarity(profile_matrix)
        sim_df = pd.DataFrame(sim_matrix, index=self.disease_profiles.index, columns=self.disease_profiles.index)
        similar = sim_df[disease_name].drop(disease_name).sort_values(ascending=False).head(n)
        return similar.reset_index().rename(columns={disease_name: 'similarity', 'index': 'disease'})

    def recommend_for_patient(self, symptom_vector, n=5):
        profile_matrix = self.disease_profiles[FEATURE_COLS].values
        patient_vec = symptom_vector[FEATURE_COLS].values
        sims = cosine_similarity(patient_vec, profile_matrix)[0]
        result = pd.DataFrame({'disease': self.disease_profiles.index, 'similarity': sims}).sort_values('similarity', ascending=False).head(n)
        return result.reset_index(drop=True)

    def predict_disease_classifier(self, symptom_vector):
        X = symptom_vector[FEATURE_COLS].values.astype('float32')
        proba = self.model.predict_proba(X)[0]
        top_idx = np.argsort(proba)[::-1][:5]
        predictions = []
        for idx in top_idx:
            disease_name = self.label_encoder.inverse_transform([idx])[0]
            predictions.append({'disease': disease_name, 'confidence': round(float(proba[idx]) * 100, 2)})
        return {'top_5': predictions, 'predicted_disease': predictions[0]['disease']}

    def get_treatment_info(self, disease_name):
        return MEDICINE_DB.get(disease_name, DEFAULT_TREATMENT)

    def recommend_full(self, symptom_dict, n=5):
        from feature_engineering import build_patient_feature_vector
        vector = build_patient_feature_vector(symptom_dict)
        content_matches = self.recommend_for_patient(vector, n=n)
        classifier_result = self.predict_disease_classifier(vector)
        top_disease = classifier_result['predicted_disease']
        treatment = self.get_treatment_info(top_disease)
        return {'content_based_matches': content_matches, 'classifier_prediction': classifier_result, 'recommended_disease': top_disease, 'treatment': treatment}
if __name__ == '__main__':
    from preprocessing import load_raw_data, clean_data
    import os
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Cleaned_Dataset.csv')
    raw = load_raw_data(data_path)
    cleaned = clean_data(raw)
    engine = MedicineRecommender(cleaned)
    print('Top 5 popular diseases:')
    print(engine.recommend_top_n_popular(5))
    print('Similar to Asthma:')
    print(engine.recommend_similar_diseases('Asthma'))
    sample = {'fever': 1, 'cough': 1, 'fatigue': 0, 'difficulty_breathing': 1, 'age': 45, 'blood_pressure': 1, 'cholesterol_level': 1}
    result = engine.recommend_full(sample)
    print(result['classifier_prediction'])
    print(result['treatment'])
