import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
    # Stable Raw URL for California Counties
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    # Note: If this fails, we fall back to a non-map visualization to prevent system crash
    try:
        response = requests.get(url)
        return response.json()
    except:
        return None

# --- 3. MAIN APP LOGIC ---
try:
    raw_df = load_and_prep_data()
    geojson = get_ca_geojson()

    # --- GLOBAL SIDEBAR FILTERS (Professor's Request) ---
    st.sidebar.title("🎛️ Intersectional Filters")
    st.sidebar.markdown("---")
    
    sel_genders = st.sidebar.multiselect("Gender", options=sorted(raw_df['GENDER'].unique()), default=raw_df['GENDER'].unique())
    sel_races = st.sidebar.multiselect("Race/Ethnicity", options=sorted(raw_df['RACE'].unique()), default=raw_df['RACE'].unique())
    sel_income = st.sidebar.multiselect("Income Tier", options=['Low', 'Middle', 'High'], default=['Low', 'Middle', 'High'])
    
    # Filter dataset globally
    df = raw_df[
        (raw_df['GENDER'].isin(sel_genders)) & 
        (raw_df['RACE'].isin(sel_races)) & 
        (raw_df['INCOME_TIER'].isin(sel_income))
    ]

    # Navigation
    page = st.sidebar.radio("Navigation", ["Overview", "Interactive Map", "Population Comparison", "Predictive Forecasting"])

    # --- TAB: OVERVIEW ---
    if page == "Overview":
        st.title("🏥 Health Equity Insights Platform (HEIP)")
        st.subheader("App Summary")
        st.markdown("""
        **What is this app?** HEIP is a strategic analytical tool designed to identify **Vertical Equity Gaps** in healthcare. It analyzes the intersection of clinical costs, personal wealth, and demographic identity to find underserved populations.

        **Target Users:**
        * **Public Health Officials:** For regional resource allocation.
        * **Policy Legislators:** To identify socio-economic groups facing extreme financial burden.

        **Key Findings:** Initial analysis indicates that **Low-Income** populations face a disproportionately high ratio of clinical expense relative to their insurance coverage.
        """)
        st.info("💡 **Instructions:** Use the sidebar to filter demographics. These settings update all charts instantly.")

    # --- TAB: INTERACTIVE MAP ---
    elif page == "Interactive Map":
        st.title("🗺️ Regional Analysis Map")
        st.markdown("> **Summary:** This map visualizes average claim costs. Darker colors represent higher financial burdens. Hover for income and insurance details.")
        
        map_stats = df.groupby('COUNTY').agg({
            'TOTAL_CLAIM_COST': 'mean',
            'INCOME': 'mean',
            'INSURANCE_COVERAGE_PCT': 'mean'
        }).reset_index()

        # If GeoJSON failed, use a bar chart; otherwise, show map
        if geojson:
            fig = px.choropleth(
                map_stats, geojson=geojson, locations='COUNTY', 
                featureidkey="properties.NAME", color='TOTAL_CLAIM_COST',
                color_continuous_scale="Viridis", scope="usa",
                hover_data={'INCOME': ':,.0f', 'INSURANCE_COVERAGE_PCT': ':.1f%'}
            )
            fig.update_geos(fitbounds="locations", visible=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Map Data Unavailable. Showing County Bar Chart instead.")
            st.bar_chart(map_stats, x='COUNTY', y='TOTAL_CLAIM_COST')

        st.divider()
        st.subheader("🔍 Deep-Dive: County Statistics")
        county = st.selectbox("Select County:", sorted(df['COUNTY'].unique()))
        c_df = df[df['COUNTY'] == county]
        
        col1, col2 = st.columns(2)
        with col1:
            dep = st.selectbox("Dependent Variable:", ['TOTAL_CLAIM_COST', 'INCOME', 'INSURANCE_COVERAGE_PCT'])
            ind = st.selectbox("Independent Variable:", ['INCOME_TIER', 'GENDER', 'RACE'])
            chart = px.bar(c_df.groupby(ind)[dep].mean().reset_index(), x=ind, y=dep, color=ind)
            st.plotly_chart(chart, use_container_width=True)

    # --- TAB: COMPARISON ---
    elif page == "Population Comparison":
        st.title("⚖️ Population Comparison")
        st.markdown("> **Summary:** Compare cost and coverage metrics across demographics side-by-side.")
        
        metric = st.selectbox("Select Metric:", ['TOTAL_CLAIM_COST', 'INCOME', 'INSURANCE_COVERAGE_PCT'])
        
        col_a, col_b = st.columns(2)
        with col_a:
            demo_a = st.selectbox("Group A by:", ['GENDER', 'RACE', 'INCOME_TIER'], key="a")
            st.plotly_chart(px.bar(df.groupby(demo_a)[metric].mean().reset_index(), x=demo_a, y=metric, color=demo_a), use_container_width=True)
        with col_b:
            demo_b = st.selectbox("Group B (Trend) by:", ['INCOME_TIER', 'RACE', 'GENDER'], key="b")
            st.plotly_chart(px.line(df.groupby(['YEAR', demo_b])[metric].mean().reset_index(), x='YEAR', y=metric, color=demo_b), use_container_width=True)

    # --- TAB: PREDICTIVE ---
    elif page == "Predictive Forecasting":
        st.title("🔮 Predictive Analytics Tool")
        st.markdown("> **Summary:** Trend projection through 2030 using Linear Regression.")
        
        target = st.selectbox("Project Target:", ['TOTAL_CLAIM_COST', 'INSURANCE_COVERAGE_PCT'])
        yearly = df.groupby('YEAR')[target].mean().reset_index()
        
        model = LinearRegression().fit(yearly[['YEAR']], yearly[target])
        future_years = np.array(range(yearly['YEAR'].max() + 1, 2031)).reshape(-1, 1)
        future_preds = model.predict(future_years)
        
        combined = pd.concat([
            pd.DataFrame({'Year': yearly['YEAR'], 'Value': yearly[target], 'Status': 'Actual'}),
            pd.DataFrame({'Year': future_years.flatten(), 'Value': future_preds, 'Status': 'Projected'})
        ])
        st.plotly_chart(px.line(combined, x='Year', y='Value', color='Status', markers=True), use_container_width=True)

except Exception as e:
    st.error("🚨 Critical System Error")
    st.exception(e)
