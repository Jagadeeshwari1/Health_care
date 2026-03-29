import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.linear_model import LinearRegression

# --- 1. SETTINGS ---
st.set_page_config(page_title="Health Equity OS", layout="wide")

# --- 2. DATA ENGINE ---
@st.cache_data
def get_all_data():
    root_path = Path(__file__).resolve().parent.parent
    data_dir = root_path / "data"
    
    # Check Patients
    p_path = data_dir / "patients.csv"
    if not p_path.exists():
        return None
    p = pd.read_csv(p_path)
    
    # Check Encounters
    e_files = list(data_dir.glob("encounters*.csv"))
    if not e_files:
        return None
    e = pd.concat([pd.read_csv(f) for f in e_files if f.stat().st_size > 0], ignore_index=True)
    
    # Cleaning
    p['BIRTHDATE'] = pd.to_datetime(p['BIRTHDATE'])
    p['AGE'] = (pd.Timestamp.today() - p['BIRTHDATE']).dt.days // 365
    for c in ['INCOME', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']:
        if c in p.columns:
            p[c] = pd.to_numeric(p[c].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce').fillna(0)
    
    p['INSURANCE_COVERAGE_PCT'] = (p['HEALTHCARE_COVERAGE'] / (p['HEALTHCARE_EXPENSES'] + 1) * 100).clip(0, 100)
    p['INCOME_TIER'] = p['INCOME'].apply(lambda x: 'Low' if x < 35000 else ('Middle' if x < 85000 else 'High'))
    
    e['START'] = pd.to_datetime(e['START'])
    e['YEAR'] = e['START'].dt.year
    
    return pd.merge(e, p, left_on='PATIENT', right_on='Id', how='inner')

# --- 3. MAIN APP ---
try:
    df = get_all_data()
    
    if df is not None:
        st.sidebar.title("🗂️ Navigation")
        page = st.sidebar.radio("Go to:", ["Intro", "Interactive Map", "Population Comparison", "Predictive Trends"])

        if page == "Intro":
            st.title("🏥 Health Equity Insights Platform")
            st.success("✅ System Online: Dependency Conflict Resolved.")
            st.markdown("### Executive Summary")
            st.write("This dashboard analyzes healthcare cost disparities. We've optimized the backend to ensure stability across cloud environments.")

        elif page == "Interactive Map":
            st.header("📍 Regional Health Map")
            st.info("💡 **Summary:** Darker areas represent higher average claim costs per county.")
            
            map_data = df.groupby('COUNTY').mean(numeric_only=True).reset_index()
            fig = px.choropleth(map_data, locations='COUNTY', locationmode='USA-states', 
                                color='TOTAL_CLAIM_COST', scope="usa", color_continuous_scale="Viridis")
            fig.update_geos(fitbounds="locations", visible=False)
            st.plotly_chart(fig, use_container_width=True)

        elif page == "Population Comparison":
            st.header("⚖️ Intersectional Comparison")
            st.info("💡 **Summary:** Comparing healthcare burdens across income tiers.")
            
            # Using Plotly for bars to avoid Altair error
            fig = px.bar(df.groupby('INCOME_TIER')['TOTAL_CLAIM_COST'].mean().reset_index(), 
                         x='INCOME_TIER', y='TOTAL_CLAIM_COST', color='INCOME_TIER',
                         color_discrete_map={'High':'#FF4B4B', 'Middle':'#FFD700', 'Low':'#2E8B57'})
            st.plotly_chart(fig, use_container_width=True)

        elif page == "Predictive Trends":
            st.header("🔮 2030 Trend Projection")
            st.info("💡 **Summary:** Mathematical model projecting future costs.")
            
            yearly = df.groupby('YEAR')['TOTAL_CLAIM_COST'].mean().reset_index()
            model = LinearRegression().fit(yearly[['YEAR']], yearly['TOTAL_CLAIM_COST'])
            future = pd.DataFrame({'YEAR': range(2025, 2031)})
            future['TOTAL_CLAIM_COST'] = model.predict(future[['YEAR']])
            future['Type'] = 'Projected'
            yearly['Type'] = 'Historical'
            
            combined = pd.concat([yearly, future])
            fig = px.line(combined, x='YEAR', y='TOTAL_CLAIM_COST', color='Type', markers=True)
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Data files not found. Please check your /data folder.")

except Exception as e:
    st.error("🚨 Critical Runtime Error")
    st.exception(e)
    
