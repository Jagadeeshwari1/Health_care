import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Try imports and show specific error if they fail
try:
    import plotly.express as px
    from sklearn.linear_model import LinearRegression
    import altair as alt
except ImportError:
    st.error("Missing Libraries! Please check requirements.txt.")
    st.stop()

# --- 1. SETTINGS ---
st.set_page_config(page_title="Health Equity OS", layout="wide")

# --- 2. FAIL-SAFE DATA ENGINE ---
@st.cache_data
def get_all_data():
    root_path = Path(__file__).resolve().parent.parent
    data_dir = root_path / "data"
    
    # 1. Load Patients
    p_path = data_dir / "patients.csv"
    if not p_path.exists():
        st.error(f"Missing patients.csv in {data_dir}")
        return None
    p = pd.read_csv(p_path)
    
    # 2. Load Encounters
    e_files = list(data_dir.glob("encounters*.csv"))
    if not e_files:
        st.error(f"No encounter files found in {data_dir}")
        return None
    e = pd.concat([pd.read_csv(f) for f in e_files if f.stat().st_size > 0], ignore_index=True)
    
    # 3. Cleaning & Metrics
    p['BIRTHDATE'] = pd.to_datetime(p['BIRTHDATE'])
    p['AGE'] = (pd.Timestamp.today() - p['BIRTHDATE']).dt.days // 365
    for c in ['INCOME', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']:
        if c in p.columns:
            p[c] = pd.to_numeric(p[c].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce').fillna(0)
    
    # Coverage % (Professor's Request)
    p['INSURANCE_COVERAGE_PCT'] = (p['HEALTHCARE_COVERAGE'] / (p['HEALTHCARE_EXPENSES'] + 1) * 100).clip(0, 100)
    p['INCOME_TIER'] = p['INCOME'].apply(lambda x: 'Low' if x < 35000 else ('Middle' if x < 85000 else 'High'))
    
    e['START'] = pd.to_datetime(e['START'])
    e['YEAR'] = e['START'].dt.year
    
    return pd.merge(e, p, left_on='PATIENT', right_on='Id', how='inner')

# --- 3. MAIN APP ---
try:
    df = get_all_data()
    
    if df is not None:
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to:", ["Intro", "Interactive Map", "Compare Populations", "Predictive Trends"])

        if page == "Intro":
            st.title("🏥 Health Equity Insights Platform")
            st.success("✅ Environment Synchronized: Data Loaded.")
            st.markdown("### Executive Summary")
            st.write("This platform analyzes **Vertical Equity** using California clinical data. Our goal is to highlight disparities in healthcare costs and insurance coverage across demographic intersections.")

        elif page == "Interactive Map":
            st.header("📍 Regional Health Map (California)")
            st.markdown("> **Summary:** This map identifies geographical cost-intensity. Darker areas represent higher average claim costs per county.")
            
            map_data = df.groupby('COUNTY').mean(numeric_only=True).reset_index()
            fig = px.choropleth(map_data, 
                                locations='COUNTY', 
                                locationmode='USA-states', 
                                color='TOTAL_CLAIM_COST', 
                                scope="usa",
                                hover_data=['INCOME', 'INSURANCE_COVERAGE_PCT'],
                                color_continuous_scale="Viridis")
            fig.update_geos(fitbounds="locations", visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Sub-county analysis
            sel_county = st.selectbox("Detailed Analysis: Select County", sorted(df['COUNTY'].unique()))
            c_df = df[df['COUNTY'] == sel_county]
            st.metric(f"{sel_county} Avg Cost", f"${c_df['TOTAL_CLAIM_COST'].mean():,.2f}")

        elif page == "Compare Populations":
            st.header("⚖️ Side-by-Side Comparison")
            st.markdown("> **Summary:** Compare healthcare burdens across different income tiers or genders to see equity gaps.")
            
            group_by = st.selectbox("Compare groups by:", ['INCOME_TIER', 'GENDER', 'RACE'])
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X(group_by, sort='-y'),
                y='mean(TOTAL_CLAIM_COST)',
                color=group_by
            ).properties(height=400)
            st.altair_chart(chart, use_container_width=True)

        elif page == "Predictive Trends":
            st.header("🔮 2030 Trend Projection")
            st.markdown("> **Summary:** We use a Linear Regression model to project healthcare cost increases over the next 5 years.")
            
            yearly = df.groupby('YEAR')['TOTAL_CLAIM_COST'].mean().reset_index()
            X = yearly[['YEAR']].values
            y = yearly['TOTAL_CLAIM_COST'].values
            
            model = LinearRegression().fit(X, y)
            future_years = np.array(range(2026, 2031)).reshape(-1, 1)
            preds = model.predict(future_years)
            
            future_df = pd.DataFrame({'YEAR': future_years.flatten(), 'TOTAL_CLAIM_COST': preds, 'Type': 'Projected'})
            yearly['Type'] = 'Historical'
            combined = pd.concat([yearly, future_df])
            
            line_chart = alt.Chart(combined).mark_line(point=True).encode(
                x='YEAR:O',
                y='TOTAL_CLAIM_COST',
                color='Type',
                strokeDash='Type'
            ).properties(height=500)
            st.altair_chart(line_chart, use_container_width=True)
            st.success(f"Projected 2030 Average Cost: **${preds[-1]:,.2f}**")

except Exception as e:
    st.error("🚨 System Initialization Error")
    st.exception(e)
    
