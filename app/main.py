import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
import requests
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
    
    # Clean Financials
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

@st.cache_data
def get_ca_geojson():
    # Fetching California County GeoJSON
    url = "https://raw.githubusercontent.com/codeforgermany/click_dummy/master/public/data/ca-counties.json"
    return requests.get(url).json()

# --- 3. MAIN APP ---
try:
    raw_df = load_and_prep_data()
    ca_geojson = get_ca_geojson()

    # --- GLOBAL SIDEBAR FILTERS ---
    st.sidebar.title("🎛️ Intersectional Filters")
    st.sidebar.info("Filters apply to all analytical pages.")
    
    selected_genders = st.sidebar.multiselect("Gender", options=sorted(raw_df['GENDER'].unique()), default=raw_df['GENDER'].unique())
    selected_races = st.sidebar.multiselect("Race/Ethnicity", options=sorted(raw_df['RACE'].unique()), default=raw_df['RACE'].unique())
    selected_income = st.sidebar.multiselect("Income Tier", options=['Low', 'Middle', 'High'], default=['Low', 'Middle', 'High'])
    
    # Apply Filters
    df = raw_df[
        (raw_df['GENDER'].isin(selected_genders)) & 
        (raw_df['RACE'].isin(selected_races)) & 
        (raw_df['INCOME_TIER'].isin(selected_income))
    ]

    # Navigation
    page = st.sidebar.radio("Navigation", ["Overview", "Interactive Map", "Population Comparison", "Predictive Forecasting"])

    # --- TAB: OVERVIEW ---
    if page == "Overview":
        st.title("🏥 Health Equity Insights Platform (HEIP)")
        st.subheader("App Summary")
        st.markdown("""
        **What is this app?** HEIP is a strategic analytical tool designed to identify **Vertical Equity Gaps** in healthcare. It analyzes the intersection of clinical costs, personal wealth, and demographic identity.

        **To whom is it useful?**
        * **Public Health Officials:** For regional resource allocation.
        * **Policy Legislators:** To identify socio-economic groups facing extreme financial burden.
        * **Clinical Researchers:** To study cost-impact across diverse populations.

        **Key Findings:** Initial analysis indicates that **Low-Income** populations face a disproportionately high ratio of clinical expense relative to their insurance coverage.
        """)
        st.info("💡 **Instructions:** Use the sidebar to filter demographics. These settings persist across all tabs.")

    # --- TAB: INTERACTIVE MAP (FIXED) ---
    elif page == "Interactive Map":
        st.title("🗺️ California Regional Analysis")
        st.markdown("> **Summary:** This choropleth map visualizes average claim costs. Darker colors represent higher financial burdens. Hover to see summary statistics.")
        
        map_stats = df.groupby('COUNTY').agg({
            'TOTAL_CLAIM_COST': 'mean',
            'INCOME': 'mean',
            'INSURANCE_COVERAGE_PCT': 'mean'
        }).reset_index()

        fig = px.choropleth(
            map_stats,
            geojson=ca_geojson,
            locations='COUNTY',
            featureidkey="properties.NAME", # Matches 'COUNTY' to GeoJSON property
            color='TOTAL_CLAIM_COST',
            color_continuous_scale="Viridis",
            hover_data={'INCOME': ':,.0f', 'INSURANCE_COVERAGE_PCT': ':.1f%'},
            labels={'TOTAL_CLAIM_COST': 'Avg Cost ($)'}
        )
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("🔍 Deep-Dive: County Statistics")
        selected_county = st.selectbox("Select a County to generate specific graphs:", sorted(df['COUNTY'].unique()))
        c_df = df[df['COUNTY'] == selected_county]
        
        col1, col2 = st.columns(2)
        with col1:
            dep_var = st.selectbox("Dependent Variable:", ['TOTAL_CLAIM_COST', 'INCOME', 'INSURANCE_COVERAGE_PCT'])
            indep_var = st.selectbox("Independent Variable:", ['INCOME_TIER', 'GENDER', 'RACE'])
            
            c_chart = px.bar(c_df.groupby(indep_var)[dep_var].mean().reset_index(), 
                             x=indep_var, y=dep_var, color=indep_var, title=f"{selected_county}: {dep_var} by {indep_var}")
            st.plotly_chart(c_chart, use_container_width=True)

    # --- TAB: COMPARISON ---
    elif page == "Population Comparison":
        st.title("⚖️ Population Comparison")
        st.markdown("> **Summary:** Compare cost and coverage metrics across demographics to identify equity gaps.")
        
        comp_metric = st.selectbox("Select Metric to Compare:", ['TOTAL_CLAIM_COST', 'INCOME', 'INSURANCE_COVERAGE_PCT'])
        
        col_a, col_b = st.columns(2)
        with col_a:
            demo_1 = st.selectbox("Compare Group A by:", ['GENDER', 'RACE', 'INCOME_TIER'], key="a")
            fig_a = px.bar(df.groupby(demo_1)[comp_metric].mean().reset_index(), x=demo_1, y=comp_metric, color=demo_1)
            st.plotly_chart(fig_a, use_container_width=True)
            
        with col_b:
            demo_2 = st.selectbox("Compare Group B by:", ['INCOME_TIER', 'RACE', 'GENDER'], key="b")
            fig_b = px.line(df.groupby(['YEAR', demo_2])[comp_metric].mean().reset_index(), 
                            x='YEAR', y=comp_metric, color=demo_2, title="Trend Lines by Population")
            st.plotly_chart(fig_b, use_container_width=True)

    # --- TAB: PREDICTIVE ---
    elif page == "Predictive Forecasting":
        st.title("🔮 Predictive Analytics Tool")
        st.markdown("> **Summary:** Trend projection through 2030 using Linear Regression.")
        
        target = st.selectbox("Select Projection Target:", ['TOTAL_CLAIM_COST', 'INCOME', 'INSURANCE_COVERAGE_PCT'])
        yearly = df.groupby('YEAR')[target].mean().reset_index()
        
        X = yearly[['YEAR']].values
        y = yearly[target].values
        model = LinearRegression().fit(X, y)
        
        future_years = np.array(range(yearly['YEAR'].max() + 1, 2031)).reshape(-1, 1)
        future_preds = model.predict(future_years)
        
        hist = pd.DataFrame({'Year': yearly['YEAR'], 'Value': y, 'Status': 'Actual'})
        proj = pd.DataFrame({'Year': future_years.flatten(), 'Value': future_preds, 'Status': 'Projected'})
        combined = pd.concat([hist, proj])
        
        fig = px.line(combined, x='Year', y='Value', color='Status', markers=True)
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"By 2030, the projected {target.lower()} is expected to reach **${future_preds[-1]:,.2f}**.")

except Exception as e:
    st.error("🚨 System Update Required. Please check file paths and columns.")
    st.exception(e)
