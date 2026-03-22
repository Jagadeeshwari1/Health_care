import streamlit as st
import pandas as pd
import glob
import os
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

# --- 1. SET PAGE CONFIG ---
st.set_page_config(page_title="Health Equity Dashboard", layout="wide", page_icon="🏥")

# --- 2. THE DATA LOGIC ---
@st.cache_data
def load_and_merge_data():
    # Find the data directory
    root_path = Path(__file__).resolve().parent.parent
    data_dir = root_path / "data"

    # Load Patients
    patients = pd.read_csv(data_dir / "patients.csv")

    # --- CRITICAL FIX: Create the 'AGE' column from 'BIRTHDATE' ---
    patients['BIRTHDATE'] = pd.to_datetime(patients['BIRTHDATE'])
    patients['AGE'] = (pd.Timestamp.today() - patients['BIRTHDATE']).dt.days // 365

    # Load Encounters
    search_pattern = str(data_dir / "encounters_part_*.csv")
    encounter_files = glob.glob(search_pattern)
    if encounter_files:
        encounters = pd.concat((pd.read_csv(f) for f in encounter_files), ignore_index=True)
    else:
        # Fallback for local testing
        encounters = pd.read_csv(data_dir / "encounters.csv") if (data_dir / "encounters.csv").exists() else pd.DataFrame()

    # Cleaning Financial Columns (Regex fix)
    for col in ['INCOME', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']:
        if col in patients.columns:
            patients[col] = pd.to_numeric(
                patients[col].astype(str).str.replace(r'[\$,]', '', regex=True), 
                errors='coerce'
            ).fillna(0)

    # Create Income Tiers
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

# --- 3. THE ML LOGIC ---
def train_prediction_model(df):
    # Features must match exactly what is in the dataframe
    features = ['AGE', 'INCOME', 'HEALTHCARE_COVERAGE']
    target = 'TOTAL_CLAIM_COST'

    # Clean and Force Numeric
    temp_df = df.copy()
    for col in features + [target]:
        temp_df[col] = pd.to_numeric(
            temp_df[col].astype(str).str.replace(r'[\$,]', '', regex=True), 
            errors='coerce'
        ).fillna(0)

    X = temp_df[features]
    y = temp_df[target]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/cost_predictor.pkl')
    return model

# --- 4. THE UI LOGIC ---
try:
    st.title("🏥 Health Equity Insights Dashboard")
    
    # Load Data
    data, report = load_and_merge_data()
    
    # Load or Train Model
    model_path = 'models/cost_predictor.pkl'
    if not os.path.exists(model_path):
        with st.spinner("Training Predictive Engine..."):
            model = train_prediction_model(data)
    else:
        model = joblib.load(model_path)

    st.success("✅ System Online: Analytics and Predictive Model Loaded.")
    
    # Display Analytics
    st.subheader("Vertical Equity Analysis: Cost Burden by City & Income")
    st.dataframe(report, use_container_width=True)

    # Simple Prediction UI
    st.divider()
    st.subheader("🔮 Individual Cost Predictor")
    c1, c2, c3 = st.columns(3)
    in_age = c1.number_input("Age", 0, 110, 45)
    in_income = c2.number_input("Income ($)", 0, 500000, 50000)
    in_cov = c3.number_input("Coverage ($)", 0, 500000, 10000)
    
    if st.button("Predict"):
        pred = model.predict([[in_age, in_income, in_cov]])[0]
        st.metric("Predicted Annual Healthcare Cost", f"${pred:,.2f}")

except Exception as e:
    st.error("🚨 Critical System Error")
    st.exception(e)
