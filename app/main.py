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

    patients = pd.read_csv(data_dir / "patients.csv")
    patients['BIRTHDATE'] = pd.to_datetime(patients['BIRTHDATE'])
    patients['AGE'] = (pd.Timestamp.today() - patients['BIRTHDATE']).dt.days // 365

    search_pattern = str(data_dir / "encounters_part_*.csv")
    encounter_files = glob.glob(search_pattern)
    encounters = pd.concat((pd.read_csv(f) for f in encounter_files), ignore_index=True) if encounter_files else pd.read_csv(data_dir / "encounters.csv")

    for col in ['INCOME', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']:
        if col in patients.columns:
            patients[col] = pd.to_numeric(patients[col].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce').fillna(0)

    # Create Tiers
    def get_tier(income):
        if income < 35000: return 'Low Income'
        if income < 85000: return 'Middle Income'
        return 'High Income'
    
    patients['INCOME_TIER'] = patients['INCOME'].apply(get_tier)
    merged_data = pd.merge(encounters, patients, left_on='PATIENT', right_on='Id')
    return merged_data

try:
    data = load_and_merge_data()
    
    # --- HEADER ---
    st.title("🏥 Strategic Health Equity Dashboard")
    st.markdown("Detailed Analysis of Financial Burden, Age Demographics, and Geographic Hotspots.")
    st.divider()

    # --- SIDEBAR ---
    st.sidebar.header("Dashboard Controls")
    selected_county = st.sidebar.multiselect("Select Counties", options=sorted(data['COUNTY'].unique()), default=data['COUNTY'].unique()[:3])
    
    # Filter Data
    view_data = data[data['COUNTY'].isin(selected_county)] if selected_county else data

    # --- ROW 1: INCOME COLORS & EXPENSIVE DISEASES ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💰 Mean Expense by Income Tier")
        income_chart_data = view_data.groupby('INCOME_TIER')['TOTAL_CLAIM_COST'].mean().reset_index()
        
        # Custom Colors for Income Tiers (Red for High Burden/Low Income)
        income_chart = alt.Chart(income_chart_data).mark_bar().encode(
            x=alt.X('INCOME_TIER', sort=['Low Income', 'Middle Income', 'High Income']),
            y='TOTAL_CLAIM_COST',
            color=alt.Color('INCOME_TIER', scale=alt.Scale(
                domain=['Low Income', 'Middle Income', 'High Income'],
                range=['#e63946', '#f1faee', '#a8dadc'] # Red, Cream, Blue
            ))
        ).properties(height=350)
        st.altair_chart(income_chart, use_container_width=True)

    with col2:
        st.subheader("⚠️ Most Expensive Health Problems")
        # Find which diseases cause the highest average expense
        expensive_diseases = view_data.groupby('DESCRIPTION')['TOTAL_CLAIM_COST'].mean().sort_values(ascending=False).head(10).reset_index()
        st.bar_chart(expensive_diseases, x="DESCRIPTION", y="TOTAL_CLAIM_COST", color="#457b9d")

    # --- ROW 2: AGE DEMOGRAPHICS & HEATMAP ---
    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("📅 Expenses by Age Group")
        # Create age bins
        view_data['AGE_GROUP'] = pd.cut(view_data['AGE'], bins=[0, 18, 35, 50, 65, 100], labels=['0-18', '19-35', '36-50', '51-65', '65+'])
        age_data = view_data.groupby('AGE_GROUP')['TOTAL_CLAIM_COST'].mean().reset_index()
        st.line_chart(age_data, x="AGE_GROUP", y="TOTAL_CLAIM_COST", color="#1d3557")

    with col4:
        st.subheader("🔥 Geographic Expense Heatmap")
        # Simple Heatmap showing Cities with highest total costs
        heatmap_data = view_data.groupby('CITY').agg({'TOTAL_CLAIM_COST': 'sum', 'Id': 'count'}).reset_index()
        
        heatmap = alt.Chart(heatmap_data).mark_rect().encode(
            x=alt.X('CITY:N', title="City"),
            y=alt.Y('TOTAL_CLAIM_COST:Q', title="Total Cumulative Cost"),
            color=alt.Color('TOTAL_CLAIM_COST:Q', scale=alt.Scale(scheme='inferno'), title="Expense Intensity"),
            tooltip=['CITY', 'TOTAL_CLAIM_COST']
        ).properties(height=350)
        st.altair_chart(heatmap, use_container_width=True)

except Exception as e:
    st.error("Dashboard Error: Reviewing component compatibility.")
    st.exception(e)
