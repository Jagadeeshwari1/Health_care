import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.linear_model import LinearRegression

# --- 1. PAGE CONFIG & SOPHISTICATED STYLING ---
st.set_page_config(page_title="Health Equity OS | NGO Insights", layout="wide", page_icon="🏥")

# Custom CSS for Big Colored Headers and Clean Layout
st.markdown("""
    <style>
    /* Main Background */
    .main {background-color: #fdfdfd;}
    
    /* Big Header Style */
    .big-header {
        background-color: #334155; 
        color: white; 
        padding: 30px; 
        border-radius: 10px; 
        font-size: 48px; 
        font-weight: 800; 
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Section Header Style */
    .section-header {
        background-color: #5eead4; 
        color: #0f172a; 
        padding: 15px; 
        border-radius: 5px; 
        font-size: 24px; 
        font-weight: 700; 
        margin-top: 20px;
        margin-bottom: 15px;
    }
    
    /* Mission Box */
    .mission-box {
        background-color: #f8fafc; 
        padding: 25px; 
        border-radius: 15px; 
        border-left: 10px solid #94a3b8; 
        margin-bottom: 25px;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #f1f5f9;
    }
    </style>
    """, unsafe_allow_html=True)

# Define NGO Soft Palette
NGO_COLORS = ["#2dd4bf", "#fb7185", "#fbbf24", "#94a3b8", "#4ade80"]

# --- 2. DATA ENGINE ---
@st.cache_data
def load_and_prep_data():
    root_path = Path(__file__).resolve().parent.parent
    data_dir = root_path / "data"
    
    p = pd.read_csv(data_dir / "patients.csv")
    p['BIRTHDATE'] = pd.to_datetime(p['BIRTHDATE'])
    p['AGE'] = (pd.Timestamp.today() - p['BIRTHDATE']).dt.days // 365
    
    for col in ['INCOME', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']:
        if col in p.columns:
            p[col] = pd.to_numeric(p[col].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce').fillna(0)
    
    p['INSURANCE_COVERAGE_PCT'] = (p['HEALTHCARE_COVERAGE'] / (p['HEALTHCARE_EXPENSES'] + 1) * 100).clip(0, 100)
    p['INCOME_TIER'] = p['INCOME'].apply(lambda x: 'Low' if x < 35000 else ('Middle' if x < 85000 else 'High'))
    
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
    sel_genders = st.sidebar.multiselect("Gender", options=sorted(raw_df['GENDER'].unique()), default=raw_df['GENDER'].unique())
    sel_races = st.sidebar.multiselect("Race/Ethnicity", options=sorted(raw_df['RACE'].unique()), default=raw_df['RACE'].unique())
    sel_income = st.sidebar.multiselect("Income Tier", options=['Low', 'Middle', 'High'], default=['Low', 'Middle', 'High'])
    
    df = raw_df[
        (raw_df['GENDER'].isin(sel_genders)) & 
        (raw_df['RACE'].isin(sel_races)) & 
        (raw_df['INCOME_TIER'].isin(sel_income))
    ]

    page = st.sidebar.radio("Navigation", ["Overview", "Interactive Map", "Population Comparison", "Predictive Forecasting"])

    # --- PAGE 1: OVERVIEW ---
    if page == "Overview":
        st.markdown('<div class="big-header">Health Equity Insights Platform</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">App Summary & Mission</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="mission-box">
            <h3 style='color: #334155;'>Strategic Mission</h3>
            HEIP identifies <strong>Vertical Equity Gaps</strong> by intersecting clinical costs with demographic identity. 
            We provide actionable data for NGOs and Public Health officials to advocate for resource redistribution.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎯 Target Stakeholders")
            st.write("* **NGO Strategists:** Regional advocacy.")
            st.write("* **Public Health Officials:** Resource planning.")
        
        with col2:
            st.subheader("📊 Key Findings")
            st.info("💡 **Insight:** Low-Income populations in the filtered cohort exhibit a significant disparity between claim costs and coverage percentage.")

    # --- PAGE 2: MAP ---
    elif page == "Interactive Map":
        st.markdown('<div class="big-header">Regional Analysis Map</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Expenditure Hotspots</div>', unsafe_allow_html=True)
        st.info("💡 **Summary:** Large bubbles represent counties with higher average healthcare costs.")
        
        map_stats = df.groupby('COUNTY').agg({
            'TOTAL_CLAIM_COST': 'mean', 'LAT': 'mean', 'LON': 'mean', 'INCOME': 'mean'
        }).reset_index()

        fig = px.scatter_mapbox(
            map_stats, lat="LAT", lon="LON", color="TOTAL_CLAIM_COST", 
            size="TOTAL_CLAIM_COST", hover_name="COUNTY",
            color_continuous_scale="Teal", size_max=25, zoom=5,
            mapbox_style="carto-positron"
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- PAGE 3: COMPARISON ---
    elif page == "Population Comparison":
        st.markdown('<div class="big-header">Intersectional Comparison</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Demographic Equity Metrics</div>', unsafe_allow_html=True)
        
        metric = st.selectbox("Metric:", ['TOTAL_CLAIM_COST', 'INSURANCE_COVERAGE_PCT'])
        
        c1, c2 = st.columns(2)
        with c1:
            demo_a = st.selectbox("Compare Group A by:", ['GENDER', 'RACE', 'INCOME_TIER'], key="a")
            st.plotly_chart(px.bar(df.groupby(demo_a)[metric].mean().reset_index(), x=demo_a, y=metric, color=demo_a, color_discrete_sequence=NGO_COLORS), use_container_width=True)
        with c2:
            demo_b = st.selectbox("Group B (Trend) by:", ['INCOME_TIER', 'RACE', 'GENDER'], key="b")
            st.plotly_chart(px.line(df.groupby(['YEAR', demo_b])[metric].mean().reset_index(), x='YEAR', y=metric, color=demo_b, color_discrete_sequence=NGO_COLORS), use_container_width=True)

    # --- PAGE 4: PREDICTIVE ---
    elif page == "Predictive Forecasting":
        st.markdown('<div class="big-header">2030 Trend Forecasting</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Predictive Analytics Tool</div>', unsafe_allow_html=True)
        
        target = st.selectbox("Project Target:", ['TOTAL_CLAIM_COST', 'INSURANCE_COVERAGE_PCT'])
        yearly = df.groupby('YEAR')[target].mean().reset_index()
        
        model = LinearRegression().fit(yearly[['YEAR']], yearly[target])
        future = pd.DataFrame({'YEAR': range(2026, 2031)})
        future[target] = model.predict(future[['YEAR']])
        
        combined = pd.concat([yearly.assign(Status='Past'), future.assign(Status='Future')])
        st.plotly_chart(px.line(combined, x='YEAR', y=target, color='Status', markers=True, color_discrete_map={'Past':'#94a3b8', 'Future':'#fb7185'}), use_container_width=True)

except Exception as e:
    st.error("🚨 System Update Required")
    st.exception(e)
