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

# --- 3. MAIN APP ---
try:
    raw_df = load_and_prep_data()

    # --- GLOBAL SIDEBAR FILTERS ---
    st.sidebar.title("🎛️ Intersectional Filters")
    st.sidebar.markdown("---")
    
    sel_genders = st.sidebar.multiselect("Gender", options=sorted(raw_df['GENDER'].unique()), default=raw_df['GENDER'].unique())
    sel_races = st.sidebar.multiselect("Race/Ethnicity", options=sorted(raw_df['RACE'].unique()), default=raw_df['RACE'].unique())
    sel_income = st.sidebar.multiselect("Income Tier", options=['Low', 'Middle', 'High'], default=['Low', 'Middle', 'High'])
    
    # Apply Filters Globally
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
        **What is this app?** HEIP is a strategic analytical tool designed to identify **Vertical Equity Gaps** in healthcare. It analyzes the intersection of clinical costs, personal wealth, and demographic identity.

        **Target Users:**
        * **Public Health Officials:** For regional resource allocation.
        * **Policy Legislators:** To identify socio-economic groups facing extreme financial burden.

        **Key Findings:** Initial analysis indicates that **Low-Income** populations face a disproportionately high ratio of clinical expense relative to their insurance coverage.
        """)
        st.info("💡 **Instructions:** Use the sidebar to filter demographics. These settings update all charts instantly.")

    # --- TAB: INTERACTIVE MAP (BUBBLE MAP VERSION) ---
    elif page == "Interactive Map":
        st.title("🗺️ Regional Analysis Map")
        st.markdown("> **Summary:** This map uses coordinate data to show healthcare spending hotspots. Larger/Darker bubbles represent higher average claim costs in that county.")
        
        # Aggregate by County and include Lat/Lon
        map_stats = df.groupby('COUNTY').agg({
            'TOTAL_CLAIM_COST': 'mean',
            'INCOME': 'mean',
            'INSURANCE_COVERAGE_PCT': 'mean',
            'LAT': 'mean',
            'LON': 'mean'
        }).reset_index()

        # Bubble Map is much more stable than Choropleth for presentations
        fig = px.scatter_mapbox(
            map_stats, lat="LAT", lon="LON", color="TOTAL_CLAIM_COST", 
            size="TOTAL_CLAIM_COST", hover_name="COUNTY",
            hover_data={'INCOME': ':,.0f', 'INSURANCE_COVERAGE_PCT': ':.1f%'},
            color_continuous_scale="Viridis", size_max=30, zoom=5,
            mapbox_style="carto-positron", title="Average Claims Cost Hotspots"
        )
        st.plotly_chart(fig, use_container_width=True)

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
    st.error("🚨 System Update in Progress. Checking data integrity...")
    st.exception(e)
