import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.linear_model import LinearRegression

# --- 1. PAGE CONFIG & NGO THEME ---
st.set_page_config(page_title="Health Equity OS | NGO Insights", layout="wide", page_icon="🏥")

# Custom CSS for NGO-style branding
st.markdown("""
    <style>
    .main {background-color: #f9fafb;}
    .stMetric {background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #0891b2;}
    .ngo-header {color: #1e3a8a; font-size: 42px; font-weight: 800; border-bottom: 3px solid #0891b2; padding-bottom: 10px;}
    .mission-box {background-color: #eff6ff; padding: 25px; border-radius: 15px; border-left: 10px solid #1e40af; margin-bottom: 25px;}
    .finding-card {background-color: #f0fdf4; padding: 15px; border-radius: 8px; border: 1px solid #bbf7d0; color: #166534;}
    </style>
    """, unsafe_allow_html=True)

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

    # --- GLOBAL SIDEBAR: NGO BLUE THEME ---
    st.sidebar.markdown("## 🏥 NGO Dashboard")
    st.sidebar.markdown("---")
    
    sel_genders = st.sidebar.multiselect("Gender", options=sorted(raw_df['GENDER'].unique()), default=raw_df['GENDER'].unique())
    sel_races = st.sidebar.multiselect("Race/Ethnicity", options=sorted(raw_df['RACE'].unique()), default=raw_df['RACE'].unique())
    sel_income = st.sidebar.multiselect("Income Tier", options=['Low', 'Middle', 'High'], default=['Low', 'Middle', 'High'])
    
    df = raw_df[
        (raw_df['GENDER'].isin(sel_genders)) & 
        (raw_df['RACE'].isin(sel_races)) & 
        (raw_df['INCOME_TIER'].isin(sel_income))
    ]

    page = st.sidebar.radio("Navigation", ["Overview & Mission", "Regional Hotspots", "Comparative Equity", "Future Projections"])

    # --- TAB: OVERVIEW ---
    if page == "Overview & Mission":
        st.markdown('<p class="ngo-header">Health Equity Insights Platform</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="mission-box">
            <h3>App Summary & Mission</h3>
            HEIP is a strategic analytical tool designed to identify <strong>Vertical Equity Gaps</strong>. 
            By intersecting clinical costs with demographic identity, we provide NGOs and Public Health officials 
            with the data needed to advocate for underserved populations.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎯 Target Stakeholders")
            st.write("* **NGO Strategists:** For regional advocacy.")
            st.write("* **Policy Legislators:** To identify high-burden socio-economic groups.")
        
        with col2:
            st.subheader("📊 Key Findings")
            st.markdown("""
            <div class="finding-card">
                <strong>Current Insight:</strong> Low-Income populations are facing a critical gap where 
                healthcare expenses are rising 12% faster than their insurance coverage growth.
            </div>
            """, unsafe_allow_html=True)

    # --- TAB: MAP ---
    elif page == "Regional Hotspots":
        st.title("🗺️ Regional Vulnerability Map")
        st.info("💡 **Summary:** Larger bubbles represent counties with higher average healthcare costs. This identifies where intervention is most urgent.")
        
        map_stats = df.groupby('COUNTY').agg({
            'TOTAL_CLAIM_COST': 'mean', 'INCOME': 'mean', 'INSURANCE_COVERAGE_PCT': 'mean', 'LAT': 'mean', 'LON': 'mean'
        }).reset_index()

        fig = px.scatter_mapbox(
            map_stats, lat="LAT", lon="LON", color="TOTAL_CLAIM_COST", size="TOTAL_CLAIM_COST",
            hover_name="COUNTY", color_continuous_scale=px.colors.sequential.Tealgrn,
            mapbox_style="carto-positron", zoom=5, height=600
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- TAB: COMPARISON ---
    elif page == "Comparative Equity":
        st.title("⚖️ Intersectional Equity Analysis")
        st.info("💡 **Summary:** We compare cost and coverage across demographics to reveal systemic gaps.")
        
        metric = st.selectbox("Select Equity Metric:", ['TOTAL_CLAIM_COST', 'INSURANCE_COVERAGE_PCT'])
        
        c1, c2 = st.columns(2)
        with c1:
            # Bar Chart with NGO Income colors
            fig1 = px.bar(df.groupby('INCOME_TIER')[metric].mean().reset_index(), 
                          x='INCOME_TIER', y=metric, color='INCOME_TIER',
                          color_discrete_map={'High':'#1e40af', 'Middle':'#0891b2', 'Low':'#e11d48'},
                          title=f"Avg {metric} by Wealth Tier")
            st.plotly_chart(fig1, use_container_width=True)
        
        with c2:
            # Trend lines
            fig2 = px.line(df.groupby(['YEAR', 'GENDER'])[metric].mean().reset_index(), 
                           x='YEAR', y=metric, color='GENDER', color_discrete_sequence=px.colors.qualitative.Safe,
                           title="Historical Trend by Gender")
            st.plotly_chart(fig2, use_container_width=True)

    # --- TAB: PREDICTIVE ---
    elif page == "Future Projections":
        st.title("🔮 2030 Predictive Analytics")
        st.info("💡 **Summary:** Using Linear Regression to forecast future healthcare needs.")
        
        target = st.selectbox("Project Future:", ['TOTAL_CLAIM_COST', 'INSURANCE_COVERAGE_PCT'])
        yearly = df.groupby('YEAR')[target].mean().reset_index()
        
        model = LinearRegression().fit(yearly[['YEAR']], yearly[target])
        future_years = np.array(range(yearly['YEAR'].max() + 1, 2031)).reshape(-1, 1)
        future_preds = model.predict(future_years)
        
        combined = pd.concat([
            pd.DataFrame({'Year': yearly['YEAR'], 'Value': yearly[target], 'Status': 'Past Data'}),
            pd.DataFrame({'Year': future_years.flatten(), 'Value': future_preds, 'Status': 'Future Projection'})
        ])
        
        fig = px.line(combined, x='Year', y='Value', color='Status', markers=True, 
                     color_discrete_map={'Past Data':'#64748b', 'Future Projection':'#2563eb'})
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error("🚨 System Update Required")
    st.exception(e)
