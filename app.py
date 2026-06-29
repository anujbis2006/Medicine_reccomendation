import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import pandas as pd
import plotly.express as px
import streamlit as st
from preprocessing import load_raw_data, clean_data, get_missing_value_report
from feature_engineering import build_disease_profiles
from recommender import MedicineRecommender
from evaluation import cross_validated_accuracy, top_k_accuracy
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'Cleaned_Dataset.csv')
st.set_page_config(page_title='Medicine Recommendation System', page_icon='💊', layout='wide')

@st.cache_data
def get_data():
    raw = load_raw_data(DATA_PATH)
    cleaned = clean_data(raw)
    return (raw, cleaned)

@st.cache_resource
def get_engine(_df):
    return MedicineRecommender(_df)
raw_df, df = get_data()
engine = get_engine(df)
st.sidebar.title('💊 MediRec')
page = st.sidebar.radio('Navigate', ['🏠 Home', '📊 Dataset Overview', '📈 Visualizations', '🎯 Recommendation Engine', '📋 Model Information', 'ℹ️ About'])
if page == '🏠 Home':
    st.title('💊 Personalized Medicine Recommendation System')
    st.write("Internship project - recommends likely diseases and basic treatment info from a patient's reported symptoms.")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Patient Records', len(df))
    col2.metric('Unique Diseases', df['disease'].nunique())
    col3.metric('Symptoms Tracked', 4)
    from medicine_db import MEDICINE_DB
    col4.metric('Diseases with Medicine Info', len(MEDICINE_DB))
    st.markdown('---')
    st.subheader('What this project does')
    st.markdown('\n        1. Takes a few symptoms + basic vitals as input\n        2. Matches them against known disease "profiles" (content-based filtering)\n        3. Also runs a classifier as a second opinion\n        4. Looks up basic medicine/advice info for the most likely disease\n\n        Use the sidebar to explore the dataset, the visualizations, or jump\n        straight to the **Recommendation Engine** page to try it out.\n        ')
elif page == '📊 Dataset Overview':
    st.title('📊 Dataset Overview')
    st.subheader('Raw sample')
    st.dataframe(raw_df.head(10), use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Shape')
        st.write(f'Rows: **{df.shape[0]}**, Columns: **{df.shape[1]}**')
        st.subheader('Missing values')
        st.dataframe(get_missing_value_report(raw_df).rename('missing_count'))
    with col2:
        st.subheader('Numeric summary')
        st.dataframe(df[['age', 'blood_pressure', 'cholesterol_level']].describe())
    with st.expander('Disease profile table (content-based feature vectors)'):
        st.dataframe(build_disease_profiles(df), use_container_width=True)
elif page == '📈 Visualizations':
    st.title('📈 Visualizations')
    st.subheader('Top 15 most common diseases')
    top_diseases = df['disease'].value_counts().head(15).reset_index()
    top_diseases.columns = ['disease', 'count']
    fig1 = px.bar(top_diseases, x='count', y='disease', orientation='h')
    fig1.update_layout(yaxis=dict(autorange='reversed'))
    st.plotly_chart(fig1, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Risk level distribution')
        risk_counts = df['risk_level'].value_counts().reset_index()
        risk_counts.columns = ['risk_level', 'count']
        fig2 = px.pie(risk_counts, names='risk_level', values='count')
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.subheader('Gender distribution')
        gender_counts = df['gender'].value_counts().reset_index()
        gender_counts.columns = ['gender', 'count']
        fig3 = px.pie(gender_counts, names='gender', values='count')
        st.plotly_chart(fig3, use_container_width=True)
    st.subheader('Age distribution')
    fig4 = px.histogram(df, x='age', nbins=20)
    st.plotly_chart(fig4, use_container_width=True)
    st.subheader('Correlation between numeric features')
    corr = df[['age', 'blood_pressure', 'cholesterol_level']].corr()
    fig5 = px.imshow(corr, text_auto=True, color_continuous_scale='Blues')
    st.plotly_chart(fig5, use_container_width=True)
elif page == '🎯 Recommendation Engine':
    st.title('🎯 Recommendation Engine')
    mode = st.radio("Choose how you'd like to get recommendations:", ['Enter symptoms (new patient)', 'Select an existing disease (find similar)'])
    if mode == 'Enter symptoms (new patient)':
        st.markdown('#### Enter patient details')
        col1, col2 = st.columns(2)
        with col1:
            fever = st.checkbox('Fever')
            cough = st.checkbox('Cough')
            fatigue = st.checkbox('Fatigue')
            breathing = st.checkbox('Difficulty breathing')
        with col2:
            age = st.slider('Age', 1, 100, 30)
            bp = st.selectbox('Blood pressure level', [0, 1, 2], format_func=lambda x: ['Low', 'Normal', 'High'][x])
            chol = st.selectbox('Cholesterol level', [0, 1, 2], format_func=lambda x: ['Low', 'Normal', 'High'][x])
        if st.button('Get Recommendation', type='primary'):
            symptom_dict = {'fever': int(fever), 'cough': int(cough), 'fatigue': int(fatigue), 'difficulty_breathing': int(breathing), 'age': age, 'blood_pressure': bp, 'cholesterol_level': chol}
            result = engine.recommend_full(symptom_dict)
            st.success(f'Most likely condition: **{result['recommended_disease']}**')
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('**Classifier top-5 predictions**')
                st.dataframe(pd.DataFrame(result['classifier_prediction']['top_5']), use_container_width=True)
            with col2:
                st.markdown('**Content-based symptom matches**')
                st.dataframe(result['content_based_matches'], use_container_width=True)
            with st.expander('💊 Suggested medicines & advice (general info only - not a substitute for a doctor)'):
                st.markdown('**Medicines**')
                for m in result['treatment']['medicines']:
                    st.write('- ' + m)
                st.markdown('**Advice**')
                for a in result['treatment']['advice']:
                    st.write('- ' + a)
            st.caption('⚠️ This is a student project for demonstration purposes only. It is not a substitute for professional medical advice.')
    else:
        disease_list = sorted(df['disease'].unique())
        chosen = st.selectbox('Select a disease', disease_list)
        if st.button('Find Similar Diseases', type='primary'):
            similar = engine.recommend_similar_diseases(chosen, n=5)
            st.markdown(f'**Diseases with a similar symptom profile to {chosen}:**')
            st.dataframe(similar, use_container_width=True)
            treatment = engine.get_treatment_info(chosen)
            with st.expander(f'💊 Medicine/advice info for {chosen}'):
                st.markdown('**Medicines**')
                for m in treatment['medicines']:
                    st.write('- ' + m)
                st.markdown('**Advice**')
                for a in treatment['advice']:
                    st.write('- ' + a)
elif page == '📋 Model Information':
    st.title('📋 Model Information')
    st.subheader('Algorithm used')
    st.markdown('\n        **Content-Based Filtering** (primary), backed by a small **TensorFlow\n        neural network** as a second opinion, plus a **popularity baseline**.\n\n        Why content-based and not collaborative filtering? Collaborative\n        filtering needs repeated user-item interactions (e.g. multiple\n        purchases/ratings per user). This dataset has none of that - just\n        one symptom record per patient, no patient IDs and no history. So\n        instead, every disease is represented by its typical symptom "profile",\n        and new patients are matched to the closest profile using cosine\n        similarity.\n        ')
    st.subheader('Features used')
    st.write('Fever, Cough, Fatigue, Difficulty breathing (as 0/1 flags), age, blood pressure level, cholesterol level')
    st.subheader('Evaluation')
    if st.button('Run evaluation (may take a few seconds)'):
        with st.spinner('Cross-validating...'):
            cv_result = cross_validated_accuracy(df)
            topk = top_k_accuracy(engine, df, k=3)
        st.json(cv_result)
        st.metric('Top-3 content-based match accuracy', topk)
    st.subheader('Limitations')
    st.markdown("\n        - Only 349 patient records spread across 116 diseases - many diseases\n          have just 1-2 examples, so accuracy is naturally limited.\n        - Only 4 symptoms are tracked; real diagnosis needs far more detail.\n        - The medicine/advice info is hand-curated for the most common\n          diseases only - rarer diseases fall back to a generic message.\n        - This tool is for learning purposes and should never replace a\n          real doctor's diagnosis.\n        ")
elif page == 'ℹ️ About':
    st.title('ℹ️ About this project')
    st.markdown('\n        **Project:** Personalized Medicine Recommendation System\n        **Type:** Internship / academic project\n        **Goal:** Practice building a small end-to-end recommendation\n        system - from raw data to a working dashboard.\n\n        Built with Python, pandas, scikit-learn, Plotly and Streamlit.\n        ')
