import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.linear_model import LinearRegression

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Health Equity OS", layout="wide", page_icon="🏥")

# --- 2. DATA ENGINE ---
@st.cache_data
def load_and_prep_data():
    root_path = Path(__file__).resolve().parent.parent
    data_dir = root_path / "data"
    
    # Load Patients
    p = pd.read_csv(data_dir / "patients.csv")
    p['BIRTHDATE'] = pd.to_datetime(p['BIRTHDATE'])
    p['AGE'] = (pd.Timestamp.today() - p['BIRTHDATE']).dt.days // 365
    
    # Financial Cleaning
    for col in ['INCOME', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']:
        if col in p.columns:
            p[col] = pd.to_numeric(p[col].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce').fillna(0)
    
    # Metrics
    p['INSURANCE_COVERAGE_PCT'] = (p['HEALTHCARE_COVERAGE'] / (p['HEALTHCARE_EXPENSES'] + 1) * 100).clip(0, 100)
    p['INCOME_TIER'] = p['INCOME'].apply(lambda x: 'Low' if x < 35000 else ('Middle' if x < 85000 else 'High'))
    
    # Load Encounters
    e_files = list(data_dir.glob("encounters*.csv"))
    e = pd.concat([pd.read_csv(f) for f in e_files if f.stat().st_size > 0], ignore_index=True)
    e['START'] = pd.to_datetime(e['START'])
    e['YEAR'] = e['START'].dt.year
    
    return pd.merge(e, p, left_on='PATIENT', right_on='Id', how='inner')

# --- 3. GLOBAL SIDEBAR FILTERS ---
def apply_sidebar_filters(df):
    st.sidebar.title("🎛️ Intersectional Filters")
    st.sidebar.markdown("---")
    
    # Demographic Sidebars (Active on every page)
    selected_genders = st.sidebar.multiselect("Gender", options=sorted(df['GENDER'].unique()), default=df['GENDER'].unique())
    selected_races = st.sidebar.multiselect("Race/Ethnicity", options=sorted(df['RACE'].unique()), default=df['RACE'].unique())
    selected_income = st.sidebar.multiselect("Income Tier", options=['Low', 'Middle', 'High'], default=['Low', 'Middle', 'High'])
    
    # Filter Logic
    mask = (df['GENDER'].isin(selected_genders)) & (df['RACE'].isin(selected_races)) & (df['INCOME_TIER'].isin(selected_income))
    return df[mask]

# --- 4. MAIN NAVIGATION & CONTENT ---
try:
    raw_df = load_and_prep_data()
    
    # Navigation Radio
    page = st.sidebar.radio("Navigation", ["Overview", "Interactive Map", "Population Comparison", "Predictive Forecasting"])
    
    # Apply global filters
    df = apply_sidebar_filters(raw_df)

    # --- TAB: OVERVIEW ---
    if page == "Overview":
        st.title("🏥 Health Equity Insights Platform (HEIP)")
        st.subheader("Project Summary")
        st.markdown("""
        **What is this app?** HEIP is a strategic analytical tool designed to identify **Vertical Equity Gaps** in healthcare. It analyzes the intersection of clinical costs, personal wealth, and demographic identity to pinpoint where healthcare resources are most needed.

        **To whom is it useful?** * **Public Health Officials:** For regional resource allocation and budget planning.
        * **Policy Legislators:** To identify socio-economic groups facing extreme financial burden.
        * **Clinical Researchers:** To study the cost-impact of chronic diseases across diverse populations.

        **Key Findings:** Initial analysis indicates that **Low-Income** populations face a disproportionately high ratio of clinical expense relative to their insurance coverage, particularly in specific urban counties.
        """)
        st.info("💡 **Instructions:** Use the sidebar to filter by Gender, Race, or Income. These filters will update all graphs in the following tabs automatically.")

    # --- TAB: MAP ---
    elif page == "Interactive Map":
        st.title("🗺️ Regional Analysis Map")
        st.markdown("> **Summary:** This map visualizes the average claim costs by county. It helps non-technical users identify high-cost 'hotspots' across the region.")
        
        map_stats = df.groupby('COUNTY').agg({
            'TOTAL_CLAIM_COST': 'mean',
            'INCOME': 'mean',
            'INSURANCE_COVERAGE_PCT': 'mean'
        }).reset_index()

        fig = px.choropleth(map_stats, locations='COUNTY', locationmode='USA-states', color='TOTAL_CLAIM_COST', 
                            scope="usa", hover_data=['INCOME', 'INSURANCE_COVERAGE_PCT'], color_continuous_scale="Viridis")
        fig.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        county = st.selectbox("Deep-Dive: Select a County", sorted(df['COUNTY'].unique()))
        c_df = df[df['COUNTY'] == county]
        st.metric(f"{county} Avg Cost", f"${c_df['TOTAL_CLAIM_COST'].mean():,.2f}")

    # --- TAB: COMPARISON ---
    elif page == "Population Comparison":
        st.title("⚖️ Side-by-Side Comparison")
        st.markdown("> **Summary:** This chart allows you to compare costs across your filtered demographics to see where the equity gaps are widest.")
        
        # side by side logic
        metric = st.selectbox("Metric to Compare:", ['TOTAL_CLAIM_COST', 'INCOME', 'INSURANCE_COVERAGE_PCT'])
        fig = px.bar(df.groupby('INCOME_TIER')[metric].mean().reset_index(), x='INCOME_TIER', y=metric, color='INCOME_TIER')
        st.plotly_chart(fig, use_container_width=True)

    # --- TAB: PREDICTIVE ---
    elif page == "Predictive Forecasting":
        st.title("🔮 2030 Predictive Tool")
        st.markdown("> **Summary:** We use mathematical trend-lines (Linear Regression) to predict how healthcare costs will rise from now until the year 2030.")
        
        target = st.selectbox("Select Projection Target:", ['TOTAL_CLAIM_COST', 'INSURANCE_COVERAGE_PCT'])
        yearly = df.groupby('YEAR')[target].mean().reset_index()
        
        # Regression
        model = LinearRegression().fit(yearly[['YEAR']], yearly[target])
        future = pd.DataFrame({'YEAR': range(2026, 2031)})
        future[target] = model.predict(future[['YEAR']])
        
        combined = pd.concat([yearly.assign(Status='Actual'), future.assign(Status='Projected')])
        fig = px.line(combined, x='YEAR', y=target, color='Status', markers=True)
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error("🚨 Configuration Error")
    st.exception(e)
