import streamlit as st
import pandas as pd
import glob
import os
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Health Equity Insights",
    page_icon="🏥",
    layout="wide"
)

# --- 2. DATA LOADING & FEATURE ENGINEERING ---
@st.cache_data
def load_and_merge_data():
    root_path = Path(__file__).resolve().parent.parent
    data_dir = root_path / "data"

    # Load Patients & Calculate Age
    patients = pd.read_csv(data_dir / "patients.csv")
    patients['BIRTHDATE'] = pd.to_datetime(patients['BIRTHDATE'])
    patients['AGE'] = (pd.Timestamp.today() - patients['BIRTHDATE']).dt.days // 365

    # Load Encounters (Handles split files)
    search_pattern = str(data_dir / "encounters_part_*.csv")
    encounter_files = glob.glob(search_pattern)
    if encounter_files:
        encounters = pd.concat((pd.read_csv(f) for f in encounter_files), ignore_index=True)
    else:
        encounters = pd.read_csv(data_dir / "encounters.csv") if (data_dir / "encounters.csv").exists() else pd.DataFrame()

    # Clean Financial Columns
    for col in ['INCOME', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']:
        if col in patients.columns:
            patients[col] = pd.to_numeric(
                patients[col].astype(str).str.replace(r'[\$,]', '', regex=True), 
                errors='coerce'
            ).fillna(0)

    # Create Income Tiers
    def get_tier(income):
        if income < 35000: return 'Low Income (<$35k)'
        if income < 85000: return 'Middle Income ($35k-$85k)'
        return 'High Income (>$85k)'
    
    patients['INCOME_TIER'] = patients['INCOME'].apply(get_tier)

    # Merge Data - Using 'Id' from patients and 'PATIENT' from encounters
    merged_data = pd.merge(encounters, patients, left_on='PATIENT', right_on='Id')
    return merged_data

# --- 3. ML TRAINING LOGIC ---
def get_model(df):
    model_path = 'models/cost_predictor.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    
    # Train if missing
    features = ['AGE', 'INCOME', 'HEALTHCARE_COVERAGE']
    target = 'TOTAL_CLAIM_COST'
    X = df[features].fillna(0)
    y = df[target].fillna(0)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, model_path)
    return model

# --- 4. DASHBOARD UI ---
try:
    data = load_and_merge_data()
    model = get_model(data)

    # --- HEADER ---
    st.title("🏥 Community Health Equity & Cost Tracker")
    st.markdown("### Analyzing Vertical Equity and Social Needs in California")
    st.divider()

    # --- SIDEBAR ---
    st.sidebar.header("🔍 Filter Analytics")
    selected_city = st.sidebar.selectbox("Select City", options=sorted(data['CITY'].unique()))
    income_filter = st.sidebar.multiselect(
        "Income Tiers", 
        options=sorted(data['INCOME_TIER'].unique()),
        default=data['INCOME_TIER'].unique()
    )

    # Apply Filters
    city_data = data[data['CITY'] == selected_city]
    filtered_data = data[data['INCOME_TIER'].isin(income_filter)]

    # --- MAIN GRAPHS ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Cost Burden by Income Tier")
        chart_data = filtered_data.groupby('INCOME_TIER')['TOTAL_CLAIM_COST'].mean().reset_index()
        st.bar_chart(chart_data, x="INCOME_TIER", y="TOTAL_CLAIM_COST", color="#ff4b4b")

    with col2:
        st.subheader(f"📍 Common Conditions in {selected_city}")
        disease_counts = city_data['DESCRIPTION'].value_counts().head(5)
        st.bar_chart(disease_counts, color="#1f77b4")

    # --- PREDICTIVE SECTION ---
    st.divider()
    st.subheader("🔮 Individual Cost Predictor")
    p1, p2, p3 = st.columns(3)
    in_age = p1.number_input("Age", 0, 110, 45)
    in_income = p2.number_input("Annual Income ($)", 0, 500000, 45000)
    in_cov = p3.number_input("Healthcare Coverage ($)", 0, 500000, 15000)

    if st.button("Generate Prediction"):
        prediction = model.predict([[in_age, in_income, in_cov]])[0]
        st.success(f"Estimated Annual Healthcare Burden: **${prediction:,.2f}**")

    # --- DATA PREVIEW (SAFE VERSION) ---
    with st.expander("🔍 View Filtered Patient Records"):
        # We only display columns that are guaranteed to exist
        available_cols = [c for c in ['CITY', 'INCOME_TIER', 'TOTAL_CLAIM_COST', 'DESCRIPTION'] if c in filtered_data.columns]
        st.dataframe(filtered_data[available_cols].head(100), use_container_width=True)

except Exception as e:
    st.error("🚨 Dashboard Error")
    st.exception(e)
