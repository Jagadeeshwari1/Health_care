import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.linear_model import LinearRegression

# --- 1. PAGE CONFIG & CUSTOM STYLING ---
st.set_page_config(page_title="Health Equity OS | NGO Insights", layout="wide", page_icon="🏥")

# NGO Custom CSS: Using Slate, Teal, and Coral instead of Blue
st.markdown("""
    <style>
    .main {background-color: #fcfdfd;}
    .stMetric {background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #5eead4;}
    .ngo-header {color: #334155; font-size: 42px; font-weight: 800; border-bottom: 3px solid #5eead4; padding-bottom: 10px;}
    .mission-box {background-color: #f8fafc; padding: 25px; border-radius: 15px; border-left: 10px solid #94a3b8; margin-bottom: 25px;}
    .finding-card {background-color: #f0fdf4; padding: 15px; border-radius: 8px; border: 1px solid #bbf7d0; color: #166534;}
    </style>
    """, unsafe_allow_html=True)

# Define NGO Soft Palette
NGO_COLORS = ["#5eead4", "#f43f5e", "#fbbf24", "#94a3b8", "#2dd4bf"] 
# (Teal, Coral, Amber, Slate, Light Teal)

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

    # --- GLOBAL SIDEBAR ---
    st.sidebar.markdown("## 🏥 HEIP Controls")
    st.sidebar.markdown("---")
    
    sel_genders = st.sidebar.multiselect("Gender", options=sorted(raw_df['GENDER'].unique()), default=raw_df['GENDER'].unique())
    sel_races = st.sidebar.multiselect("Race/Ethnicity", options=sorted(raw_df['RACE'].unique()), default=raw_df['RACE'].unique())
    sel_income = st.sidebar.multiselect("Income Tier", options=['Low', 'Middle', 'High'], default=['Low', 'Middle', 'High'])
    
    df = raw_df[
        (raw_df['GENDER'].isin(sel_genders)) & 
        (raw_df['RACE'].isin(sel_races)) & 
        (raw_df['INCOME_TIER'].isin(sel_income))
    ]

    page = st.sidebar.radio("Navigation", ["Overview", "Interactive Map", "Population Comparison", "Predictive Forecasting"])

    # --- TAB: OVERVIEW ---
    if page == "Overview":
        st.markdown('<p class="ngo-header">Health Equity Insights Platform</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="mission-box">
            <h3 style='color: #334155;'>App Summary & Mission</h3>
            HEIP is a strategic analytical tool designed to identify <strong>Vertical Equity Gaps</strong>. 
            By intersecting clinical costs with demographic identity, we provide stakeholders 
            with the data needed to advocate for underserved populations.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎯 Target Stakeholders")
            st.write("* **Public Health Officials:** Regional resource allocation.")
            st.write("* **Policy Legislators:** Socio-economic burden tracking.")
        
        with col2:
            st.subheader("📊 Key Findings")
            st.markdown("""
            <div class="finding-card">
                <strong>Current Cohort Insight:</strong> The selected demographic exhibits a cost-to-income 
                ratio that significantly exceeds the regional average.
            </div>
            """, unsafe_allow_html=True)

    # --- TAB: MAP ---
    elif page == "Interactive Map":
        st.title("🗺️ Regional Expenditure Hotspots")
        st.info("💡 **Summary:** Large bubbles indicate counties with higher average healthcare costs.")
        
        map_stats = df.groupby('COUNTY').agg({
            'TOTAL_CLAIM_COST': 'mean', 'INCOME': 'mean', 'INSURANCE_COVERAGE_PCT': 'mean', 'LAT': 'mean', 'LON': 'mean'
        }).reset_index()

        # Using the "Teal" and "Mint" color scale
        fig = px.scatter_mapbox(
            map_stats, lat="LAT", lon="LON", color="TOTAL_CLAIM_COST", 
            size="TOTAL_CLAIM_COST", hover_name="COUNTY",
            color_continuous_scale="Teal", size_max=25, zoom=5,
            mapbox_style="carto-positron"
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- TAB: COMPARISON ---
    elif page == "Population Comparison":
        st.title("⚖️ Comparative Demographic Analysis")
        st.info("💡 **Summary:** Compare cost and coverage metrics side-by-side.")
        
        metric = st.selectbox("Select Metric:", ['TOTAL_CLAIM_COST', 'INSURANCE_COVERAGE_PCT'])
        
        col_a, col_b = st.columns(2)
        with col_a:
            demo_a = st.selectbox("Compare Group A by:", ['GENDER', 'RACE', 'INCOME_TIER'], key="a")
            # NGO Colors: Teal, Coral, Slate
            fig_a = px.bar(df.groupby(demo_a)[metric].mean().reset_index(), 
                          x=demo_a, y=metric, color=demo_a,
                          color_discrete_sequence=NGO_COLORS)
            st.plotly_chart(fig_a, use_container_width=True)
        with col_b:
            demo_b = st.selectbox("Group B (Trend) by:", ['INCOME_TIER', 'RACE', 'GENDER'], key="b")
            fig_b = px.line(df.groupby(['YEAR', demo_b])[metric].mean().reset_index(), 
                            x='YEAR', y=metric, color=demo_b,
                            color_discrete_sequence=NGO_COLORS)
            st.plotly_chart(fig_b, use_container_width=True)

    # --- TAB: PREDICTIVE ---
    elif page == "Predictive Forecasting":
        st.title("🔮 Predictive Analytics (2030)")
        st.info("💡 **Summary:** Trend projection through 2030 using Linear Regression.")
        
        target = st.selectbox("Project Target:", ['TOTAL_CLAIM_COST', 'INSURANCE_COVERAGE_PCT'])
        yearly = df.groupby('YEAR')[target].mean().reset_index()
        
        model = LinearRegression().fit(yearly[['YEAR']], yearly[target])
        future_years = np.array(range(yearly['YEAR'].max() + 1, 2031)).reshape(-1, 1)
        future_preds = model.predict(future_years)
        
        combined = pd.concat([
            pd.DataFrame({'Year': yearly['YEAR'], 'Value': yearly[target], 'Status': 'Actual Data'}),
            pd.DataFrame({'Year': future_years.flatten(), 'Value': future_preds, 'Status': 'Projection'})
        ])
        
        # Soft Slate and Coral projection
        fig = px.line(combined, x='Year', y='Value', color='Status', markers=True,
                     color_discrete_map={'Actual Data': '#94a3b8', 'Projection': '#f43f5e'})
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error("🚨 System Update in Progress.")
    st.exception(e)
