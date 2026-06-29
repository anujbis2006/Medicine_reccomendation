# Personalized Medicine Recommendation System

An internship project that recommends a likely disease and basic
medicine/advice information based on a patient's symptoms and vitals.

## Project Overview

This started as a "personalized medicine recommendation" idea and went
through a few different directions (a Flask app, some exploratory
notebooks) before settling on the approach in this repo: a clean,
content-based recommendation engine wrapped in a Streamlit dashboard.

## Business Problem

Given a patient's symptoms (fever, cough, fatigue, difficulty breathing)
and a few vitals (age, blood pressure level, cholesterol level), suggest
the most likely condition and point them toward general medicine/advice
information for that condition. This is meant as a learning project, not
a diagnostic tool.

## Dataset Description

`data/Cleaned_Dataset.csv` - 349 patient records (300 after removing exact
duplicates), each with:

| Column | Description |
|---|---|
| `disease` | Diagnosed condition (116 unique diseases) |
| `fever`, `cough`, `fatigue`, `difficulty_breathing` | Yes/No symptom flags |
| `age` | Patient age |
| `gender` | male / female |
| `blood_pressure`, `cholesterol_level` | Encoded levels (0/1/2) |
| `age_scaled`, `bp_scaled`, `chol_scaled` | Pre-scaled numeric versions |
| `risk_level` | Low / Medium / High |
| `outcome_variable` | Positive / Negative |

There is **no user ID and no repeated interaction history** in this data -
each row is a single, independent patient record. That fact directly
shaped the choice of recommendation approach below.

## Why Content-Based Filtering?

Collaborative filtering (the "users like you also liked..." approach)
needs a user-item interaction matrix - multiple ratings/purchases per
user. This dataset doesn't have that: there are no patient IDs and no
repeated visits, just one symptom snapshot per row.

What the data *does* give us is a clear symptom "profile" for each
disease. So the recommender:

1. Builds a feature vector for every disease (the average symptom pattern
   + vitals of patients who had it).
2. Builds the same kind of vector for a new patient's input.
3. Uses cosine similarity to find the closest matching disease profile(s)
   - this is the content-based recommendation.
4. Backs this up with a small TensorFlow neural network trained on the same
   features, and a popularity baseline (most common diseases overall).
5. Looks up curated medicine/advice info for the top match.

This is a realistic, appropriately-scoped approach for the size and shape
of this dataset - a deep learning or full collaborative filtering model
would be overkill and wouldn't have the right data to learn from anyway.

## Folder Structure

```
project/
│
├── data/                       # the cleaned dataset
├── notebooks/
│   └── 01_eda.ipynb             # exploratory data analysis
├── src/
│   ├── preprocessing.py         # load + clean the data
│   ├── feature_engineering.py   # build disease profiles / patient vectors
│   ├── recommender.py           # the recommendation engine itself
│   ├── evaluation.py            # cross-validation + top-k accuracy
│   ├── medicine_db.py           # curated medicine/advice lookup table
│   └── utils.py
├── models/
│   └── recommender.pkl          # a pre-trained, pickled recommender
├── dashboard/                   # (kept for future dashboard assets/screenshots)
├── app.py                       # Streamlit dashboard entry point
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv venv
source venv/bin/activate      # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Running the Dashboard

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## Model Explanation

- **Content-based filtering** (main approach): cosine similarity between
  a patient's symptom/vitals vector and each disease's average profile.
- **TensorFlow neural network**: a small 2-hidden-layer dense network
  (32-32 units) trained on the same features, used as a second prediction
  and for showing top-5 candidate diseases with confidence scores.
- **Popularity baseline**: simple "most common diseases" list, used for
  the Top-N view and as a sensible fallback.
- Unseen/rare diseases and unusual symptom combinations are handled
  gracefully - the system falls back to the popularity list or a generic
  "consult a doctor" message rather than failing.

## Results

- Stratified 3-fold cross-validation accuracy on the classifier: roughly
  **8-10%** when restricted to diseases with enough samples per fold.
- Top-3 content-based match accuracy (true disease appears in the top 3
  similarity matches): roughly **35%**.

These numbers are modest, and that's expected and worth being upfront
about - see Limitations below.

## Limitations

- Only 300-349 records spread across 116 diseases means most diseases
  have just 1-3 examples. There isn't enough data to learn fine-grained
  patterns reliably.
- Only 4 symptoms are tracked. Real diagnosis uses far more information
  (lab results, full symptom history, imaging, etc.).
- The medicine/advice database is hand-curated for ~10 common diseases.
  Everything else falls back to a generic message.
- This is an educational project. It should never be used as a
  substitute for seeing an actual doctor.

## Future Improvements

- Collect more records per disease (or fewer, broader disease categories)
  to make the classifier meaningfully more accurate.
- Track more symptoms/lab values as features.
- Expand the medicine/advice database to cover more conditions.
- If real patient interaction history ever becomes available (e.g.
  repeat visits, treatment outcomes), revisit collaborative filtering or
  a hybrid approach on top of this content-based baseline.
