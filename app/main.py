import streamlit as st
import pandas as pd
import glob
import os
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import altair as alt

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Health Equity Analytics", layout="wide", page_icon="🏥")

@st.cache_data
def load_and_merge_data():
    root_path = Path(__file__).resolve().parent.parent
    data_dir = root_path / "data"

    # Load Patients
    patients = pd.read_csv(data_dir / "patients.csv")
    patients['BIRTHDATE'] = pd.to_datetime(patients['BIRTHDATE'])
    patients['AGE'] = (pd.Timestamp.today() - patients['BIRTHDATE']).dt.days // 365

    # Load Encounters
    search_pattern = str(data_dir / "encounters_part_*.csv")
    encounter_files = glob.glob(search_pattern)
    if encounter_files:
        encounters = pd.concat((pd.read_csv(f) for f in encounter_files), ignore_index=True)
    else:
        encounters = pd.read_csv(data_dir / "encounters.csv") if (data_dir / "encounters.csv").exists() else pd.DataFrame()

    # Clean Financials
    for col in ['INCOME', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']:
        if col in patients.columns:
            patients[col] = pd.to_numeric(patients[col].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce').fillna(0)

    # Create Tiers
    def get_tier(income):
        if income < 35000: return 'Low Income'
        if income < 85000: return 'Middle Income'
        return 'High Income'
    
    patients['INCOME_TIER'] = patients['INCOME'].apply(get_tier)
    
    # Merge - We keep all columns from both tables
    merged_data = pd.merge(encounters, patients, left_on='PATIENT', right_on='Id', how='inner')
    return merged_data

try:
    data = load_and_merge_data()
    
    # --- HEADER ---
    st.title("🏥 Strategic Health Equity Dashboard")
    st.markdown("Analyzing Financial Burden, Disease Cost-Impact, and Geographic Hotspots.")
    st.divider()

    # --- SIDEBAR ---
    st.sidebar.header("Dashboard Filters")
    # Using a safer way to get unique counties
    available_counties = sorted(data['COUNTY'].unique()) if 'COUNTY' in data.columns else []
    selected_county = st.sidebar.multiselect("Select Counties", options=available_counties, default=available_counties[:5])
    
    # Filter Data
    view_data = data[data['COUNTY'].isin(selected_county)] if selected_county else data

    # --- ROW 1: INCOME COLORS & EXPENSIVE DISEASES ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💰 Mean Expense by Income Tier")
        income_chart_data = view_data.groupby('INCOME_TIER')['TOTAL_CLAIM_COST'].mean().reset_index()
        
        income_chart = alt.Chart(income_chart_data).mark_bar().encode(
            x=alt.X('INCOME_TIER', sort=['Low Income', 'Middle Income', 'High Income'], title="Economic Status"),
            y=alt.Y('TOTAL_CLAIM_COST', title="Average Cost ($)"),
            color=alt.Color('INCOME_TIER', scale=alt.Scale(
                domain=['Low Income', 'Middle Income', 'High Income'],
                range=['#e63946', '#f1faee', '#a8dadc'] # Red for High Burden
            ), legend=None)
        ).properties(height=350)
        st.altair_chart(income_chart, use_container_width=True)

    with col2:
        st.subheader("⚠️ High-Cost Clinical Conditions")
        # Top 10 expensive diseases by mean cost
        expensive_diseases = view_data.groupby('DESCRIPTION')['TOTAL_CLAIM_COST'].mean().sort_values(ascending=False).head(10).reset_index()
        st.bar_chart(expensive_diseases, x="DESCRIPTION", y="TOTAL_CLAIM_COST", color="#457b9d")

    # --- ROW 2: AGE DEMOGRAPHICS & HEATMAP ---
    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("📅 Expense Trend by Age Group")
        # Create age groups
        view_data['AGE_GROUP'] = pd.cut(view_data['AGE'], bins=[0, 18, 35, 50, 65, 120], labels=['0-18', '19-35', '36-50', '51-65', '65+'])
        age_data = view_data.groupby('AGE_GROUP', observed=False)['TOTAL_CLAIM_COST'].mean().reset_index()
        st.line_chart(age_data, x="AGE_GROUP", y="TOTAL_CLAIM_COST", color="#1d3557")

    with col4:
        st.subheader("🔥 Geographic Expense Intensity (Heatmap)")
        # THE FIX: Using 'CITY' for count instead of 'Id' to avoid KeyError
        heatmap_data = view_data.groupby('CITY').agg({'TOTAL_CLAIM_COST': 'sum', 'CITY': 'count'}).rename(columns={'CITY': 'PatientCount'}).reset_index()
        
        heatmap = alt.Chart(heatmap_data).mark_rect().encode(
            x=alt.X('CITY:N', title="City"),
            y=alt.Y('TOTAL_CLAIM_COST:Q', title="Cumulative Expense"),
            color=alt.Color('TOTAL_CLAIM_COST:Q', scale=alt.Scale(scheme='inferno'), title="Intensity"),
            tooltip=['CITY', 'TOTAL_CLAIM_COST', 'PatientCount']
        ).properties(height=350)
        st.altair_chart(heatmap, use_container_width=True)

except Exception as e:
    st.error("🚨 Dashboard Component Error")
    st.exception(e)
