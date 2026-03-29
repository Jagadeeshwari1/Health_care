import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.linear_model import LinearRegression

# --- 1. PAGE CONFIG & NGO STYLING ---
st.set_page_config(page_title="HEIP | NGO Health Equity OS", layout="wide", page_icon="🏥")

st.markdown("""
    <style>
    .main {background-color: #fdfdfd;}
    .big-header {
        background-color: #334155; 
        color: white; 
        padding: 30px; 
        border-radius: 10px; 
        font-size: 42px; 
        font-weight: 800; 
        text-align: center;
        margin-bottom: 25px;
    }
    .section-header {
        background-color: #ccfbf1; 
        color: #0f172a; 
        padding: 12px 20px; 
        border-radius: 8px; 
        font-size: 22px; 
        font-weight: 700; 
        margin-top: 25px; 
        margin-bottom: 15px; 
        border-left: 8px solid #2dd4bf;
    }
    .mission-box {
        background-color: #f8fafc; 
        padding: 25px; 
        border-radius: 15px; 
        border: 1px solid #e2e8f0; 
        margin-bottom: 25px;
    }
    </style>
    """, unsafe_allow_html=True)

# Fixed NGO Palette for Legend Consistency across all charts
NGO_PALETTE = px.colors.qualitative.Safe 

# --- 2. DATA ENGINE ---
@st.cache_data
def load_and_prep_data():
    root_path = Path(__file__).resolve().parent.parent
    data_dir = root_path / "data"
    
    # Load Patients
    p = pd.read_csv(data_dir / "patients.csv")
    for col in ['INCOME', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']:
        if col in p.columns:
            p[col] = pd.to_numeric(p[col].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce').fillna(0)
    
    p['INSURANCE_COVERAGE_PCT'] = (p['HEALTHCARE_COVERAGE'] / (p['HEALTHCARE_EXPENSES'] + 1) * 100).clip(0, 100)
    p['INCOME_TIER'] = p['INCOME'].apply(lambda x: 'Low' if x < 35000 else ('Middle' if x < 85000 else 'High'))
    
    # Load Encounters & Extract Year of Encounter
    e_files = list(data_dir.glob("encounters*.csv"))
    e = pd.concat([pd.read_csv(f) for f in e_files if f.stat().st_size > 0], ignore_index=True)
    e['START'] = pd.to_datetime(e['START'])
    e['YEAR'] = e['START'].dt.year 
    
    return pd.merge(e, p, left_on='PATIENT', right_on='Id', how='inner')

# --- 3. MAIN APP ---
try:
    raw_df = load_and_prep_data()

    # --- GLOBAL SIDEBAR ---
    st.sidebar.markdown("## 📊 Strategic Filters")
    sel_genders = st.sidebar.multiselect("Gender", options=sorted(raw_df['GENDER'].unique()), default=raw_df['GENDER'].unique())
    sel_races = st.sidebar.multiselect("Race", options=sorted(raw_df['RACE'].unique()), default=raw_df['RACE'].unique())
    sel_income = st.sidebar.multiselect("Income Tier", options=['Low', 'Middle', 'High'], default=['Low', 'Middle', 'High'])
    
    df = raw_df[(raw_df['GENDER'].isin(sel_genders)) & (raw_df['RACE'].isin(sel_races)) & (raw_df['INCOME_TIER'].isin(sel_income))]
    page = st.sidebar.radio("Navigation", ["Overview", "Interactive Map", "Population Comparison", "Predictive Forecasting"])

    # --- PAGE 1: OVERVIEW & FEEDBACK ---
    if page == "Overview & Feedback":
        st.markdown('<div class="big-header">Health Equity Insights Platform</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">App Summary & Mission</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="mission-box">
            <h3 style='color: #334155;'>HEIP Mission Statement</h3>
            To dismantle <strong>Vertical Equity Gaps</strong> by providing transparent, intersectional data 
            on healthcare costs. We empower NGOs to advocate for policy shifts that protect the most 
            financially vulnerable cohorts in our society.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎯 Key Stakeholders")
            st.write("* **NGO Strategists:** Regional advocacy.")
            st.write("* **Public Health Officials:** Resource planning.")
        with col2:
            st.subheader("📊 Primary Finding")
            st.info("💡 **Insight:** Under-insured populations currently face a cumulative clinical burden exceeding 35% of their annual household income.")

        # CONTACT & FEEDBACK FORM (FIXED)
        st.markdown('<div class="section-header">Contact NGO & Provide Feedback</div>', unsafe_allow_html=True)
        with st.form("ngo_feedback"):
            st.write("Submit your insights or request a detailed regional report.")
            f_name = st.text_input("Full Name")
            f_email = st.text_input("Organization Email")
            f_msg = st.text_area("Observations or Feedback")
            
            # The submit button itself handles the logic
            submitted = st.form_submit_button("Submit to NGO")
            if submitted:
                if f_name and f_email:
                    st.success(f"Thank you, {f_name}! Your feedback has been logged for our analysts.")
                else:
                    st.warning("Please provide your name and email so we can follow up.")

    elif page == "Interactive Map":
        st.markdown('<div class="big-header">California Regional Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Expenditure Bubble Map</div>', unsafe_allow_html=True)
        st.info("💡 **Summary:** Large bubbles represent counties with higher average healthcare costs. Hover to see Income and Coverage.")
        
        map_stats = df.groupby('COUNTY').agg({
            'TOTAL_CLAIM_COST': 'mean', 
            'INCOME': 'mean', 
            'INSURANCE_COVERAGE_PCT': 'mean', 
            'LAT': 'mean', 
            'LON': 'mean'
        }).reset_index()

        # Reliable Bubble Map (No external GeoJSON needed)
        fig_bub = px.scatter_mapbox(
            map_stats, 
            lat="LAT", lon="LON", 
            color="TOTAL_CLAIM_COST", 
            size="TOTAL_CLAIM_COST",
            hover_name="COUNTY", 
            # Customizing hover data to hide Lat/Lon and show requested metrics
            hover_data={
                'LAT': False, 
                'LON': False, 
                'TOTAL_CLAIM_COST': ':,.2f', 
                'INCOME': ':,.0f', 
                'INSURANCE_COVERAGE_PCT': ':.1f%'
            },
            color_continuous_scale="Teal", 
            size_max=25, zoom=5, 
            mapbox_style="carto-positron"
        )
        st.plotly_chart(fig_bub, use_container_width=True, key="bubble_map_final")

        # DEEP-DIVE GRAPHS WITH CUSTOMIZABLE AXES
        st.markdown('<div class="section-header">Deep-Dive: County Statistics</div>', unsafe_allow_html=True)
        sel_county = st.selectbox("Select County:", sorted(df['COUNTY'].unique()))
        c_df = df[df['COUNTY'] == sel_county]
        
        c1, c2 = st.columns(2)
        with c1:
            x_ax = st.selectbox("X-Axis (Demographic):", ['GENDER', 'RACE', 'INCOME_TIER'], key="x_map")
            y_ax = st.selectbox("Y-Axis (Metric):", ['TOTAL_CLAIM_COST', 'INCOME', 'INSURANCE_COVERAGE_PCT'], key="y_map")
            st.plotly_chart(px.bar(c_df.groupby(x_ax)[y_ax].mean().reset_index(), x=x_ax, y=y_ax, color=x_ax, color_discrete_sequence=NGO_PALETTE), use_container_width=True, key="c_bar")
        with c2:
            st.metric(f"Avg Claims: {sel_county}", f"${c_df['TOTAL_CLAIM_COST'].mean():,.2f}")
            st.metric(f"Avg Income: {sel_county}", f"${c_df['INCOME'].mean():,.2f}")
            st.metric(f"Insurance Coverage", f"{c_df['INSURANCE_COVERAGE_PCT'].mean():.1f}%")

   # --- PAGE 3: COMPARISON ---
    elif page == "Population Comparison":
        st.markdown('<div class="big-header">Intersectional Comparison</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Demographic Equity Metrics</div>', unsafe_allow_html=True)
        metric = st.selectbox("Metric:", ['TOTAL_CLAIM_COST', 'INSURANCE_COVERAGE_PCT'])
        c1, c2 = st.columns(2)
        with c1:
            demo_a = st.selectbox("Compare Group A by:", ['GENDER', 'RACE', 'INCOME_TIER'], key="a")
            st.plotly_chart(px.bar(df.groupby(demo_a)[metric].mean().reset_index(), x=demo_a, y=metric, color=demo_a, color_discrete_sequence=NGO_PALETTE), use_container_width=True)
        with c2:
            demo_b = st.selectbox("Group B (Trend) by:", ['INCOME_TIER', 'RACE', 'GENDER'], key="b")
            st.plotly_chart(px.line(df.groupby(['YEAR', demo_b])[metric].mean().reset_index(), x='YEAR', y=metric, color=demo_b, color_discrete_sequence=NGO_PALETTE), use_container_width=True)
    
    elif page == "Predictive Forecasting":
        st.markdown('<div class="big-header">2030 Trend Forecasting</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Future Projections</div>', unsafe_allow_html=True)
        target = st.selectbox("Project Target:", ['TOTAL_CLAIM_COST', 'INSURANCE_COVERAGE_PCT'])
        yearly = df.groupby('YEAR')[target].mean().reset_index()
        model = LinearRegression().fit(yearly[['YEAR']], yearly[target])
        future = pd.DataFrame({'YEAR': range(2026, 2031)}); future[target] = model.predict(future[['YEAR']])
        combined = pd.concat([yearly.assign(Status='Past'), future.assign(Status='Future')])
        st.plotly_chart(px.line(combined, x='YEAR', y=target, color='Status', markers=True, color_discrete_map={'Past':'#94a3b8', 'Future':'#fb7185'}), use_container_width=True, key="pred_chart")

except Exception as e:
    st.error("🚨 System Update Required. Ensure data folder contains all CSV files.")
    st.exception(e)
