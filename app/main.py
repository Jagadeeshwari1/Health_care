import streamlit as st
import pandas as pd
import glob
import os
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

# --- 1. SET PAGE CONFIG ---
st.set_page_config(page_title="Health Equity Dashboard", layout="wide", page_icon="🏥")

# --- 2. THE DATA LOGIC (Previously in src/data_processor.py) ---
@st.cache_data
def load_and_merge_data():
    # Find the data directory
    root_path = Path(__file__).resolve().parent.parent
    data_dir = root_path / "data"

    # Load Patients
    patients = pd.read_csv(data_dir / "patients.csv")

    # Load Encounters (Handles split files)
    search_pattern = str(data_dir / "encounters_part_*.csv")
    encounter_files = glob.glob(search_pattern)
    encounters = pd.concat((pd.read_csv(f) for f in encounter_files), ignore_index=True)

    # Cleaning Financial Columns (Regex fix)
    for col in ['INCOME', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']:
        if col in patients.columns:
            patients[col] = pd.to_numeric(
                patients[col].astype(str).str.replace(r'[\$,]', '', regex=True), 
                errors='coerce'
            ).fillna(0)

    # Create Income Tiers for Vertical Equity Analysis
    patients['INCOME_TIER'] = patients['INCOME'].apply(
        lambda x: 'Low Income' if x < 35000 else ('Middle Income' if x < 85000 else 'High Income')
    )

    # Merge
    merged_data = pd.merge(encounters, patients, left_on='PATIENT', right_on='Id')

    # Create the Report
    report = merged_data.groupby(['CITY', 'INCOME_TIER']).agg({
        'TOTAL_CLAIM_COST': 'mean',
        'HEALTHCARE_EXPENSES': 'mean'
    }).reset_index()

    return merged_data, report

# --- 3. THE ML LOGIC (Previously in src/model.py) ---
def train_prediction_model(df):
    features = ['AGE', 'INCOME', 'HEALTHCARE_COVERAGE']
    target = 'TOTAL_CLAIM_COST'

    # Clean and Force Numeric
    for col in features + [target]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(r'[\$,]', '', regex=True), 
            errors='coerce'
        ).fillna(0)

    X = df[features]
    y = df[target]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save locally in the container
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/cost_predictor.pkl')
    return model

# --- 4. THE UI LOGIC ---
try:
    st.title("🏥 Health Equity Insights Dashboard")
    
    # Load Data
    data, report = load_and_merge_data()
    
    # Load or Train Model
    if not os.path.exists('models/cost_predictor.pkl'):
        with st.spinner("Training Predictive Engine..."):
            model = train_prediction_model(data)
    else:
        model = joblib.load('models/cost_predictor.pkl')

    st.success("✅ System Online: Analytics and Predictive Model Loaded.")
    
    # Display Analytics
    st.subheader("Vertical Equity Analysis: Cost Burden by City & Income")
    st.dataframe(report, use_container_width=True)

except Exception as e:
    st.error("🚨 Critical System Error")
    st.exception(e)
    st.info("Check that 'patients.csv' and 'encounters' are in the /data folder on GitHub.")
