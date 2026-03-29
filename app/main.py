import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Try imports and show specific error if they fail
try:
    import plotly.express as px
    from sklearn.linear_model import LinearRegression
except ImportError:
    st.error("Missing Libraries! Please ensure 'plotly' and 'scikit-learn' are in your requirements.txt file.")
    st.stop()

# --- 1. SETTINGS ---
st.set_page_config(page_title="Health Equity OS", layout="wide")

# --- 2. FAIL-SAFE DATA ENGINE ---
@st.cache_data
def get_all_data():
    root_path = Path(__file__).resolve().parent.parent
    data_dir = root_path / "data"
    
    # Check if directory exists
    if not data_dir.exists():
        st.error(f"Data folder not found at {data_dir}")
        return None

    # Load Patients
    p_path = data_dir / "patients.csv"
    if not p_path.exists():
        st.error(f"Missing patients.csv in {data_dir}")
        return None
    p = pd.read_csv(p_path)
    
    # Load Encounters
    e_files = list(data_dir.glob("encounters*.csv"))
    if not e_files:
        st.error(f"No encounter files found in {data_dir}")
        return None
    
    e = pd.concat([pd.read_csv(f) for f in e_files if f.stat().st_size > 0], ignore_index=True)
    
    # Cleaning
    p['BIRTHDATE'] = pd.to_datetime(p['BIRTHDATE'])
    p['AGE'] = (pd.Timestamp.today() - p['BIRTHDATE']).dt.days // 365
    for c in ['INCOME', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']:
        if c in p.columns:
            p[c] = pd.to_numeric(p[c].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce').fillna(0)
    
    p['INSURANCE_COVERAGE_PCT'] = (p['HEALTHCARE_COVERAGE'] / (p['HEALTHCARE_EXPENSES'] + 1) * 100).clip(0, 100)
    
    e['START'] = pd.to_datetime(e['START'])
    e['YEAR'] = e['START'].dt.year
    
    return pd.merge(e, p, left_on='PATIENT', right_on='Id', how='inner')

# --- 3. MAIN APP ---
try:
    df = get_all_data()
    
    if df is not None:
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to:", ["Intro", "Interactive Map", "Compare Groups", "Predictive Tool"])

        if page == "Intro":
            st.title("🏥 Health Equity Insights")
            st.success("System Online: Data Loaded Successfully.")
            st.write("Use the sidebar to explore the findings requested by the professor.")

        elif page == "Interactive Map":
            st.header("📍 Interactive Map")
            st.info("💡 **Summary:** Visualizing clinical burden across California counties.")
            fig = px.choropleth(df.groupby('COUNTY').mean(numeric_only=True).reset_index(), 
                                locations='COUNTY', locationmode='USA-states', color='TOTAL_CLAIM_COST', scope="usa")
            fig.update_geos(fitbounds="locations", visible=False)
            st.plotly_chart(fig, use_container_width=True)

        elif page == "Compare Groups":
            st.header("⚖️ Group Comparison")
            st.info("💡 **Summary:** Side-by-side demographic analysis.")
            metric = st.selectbox("Metric", ['TOTAL_CLAIM_COST', 'INCOME'])
            st.bar_chart(df.groupby('INCOME_TIER')[metric].mean())

        elif page == "Predictive Tool":
            st.header("🔮 Future Projections")
            st.info("💡 **Summary:** 2030 Trend Forecasting.")
            # Simple Regression logic
            yearly = df.groupby('YEAR')['TOTAL_CLAIM_COST'].mean().reset_index()
            model = LinearRegression().fit(yearly[['YEAR']], yearly['TOTAL_CLAIM_COST'])
            future = pd.DataFrame({'YEAR': range(2026, 2031)})
            future['TOTAL_CLAIM_COST'] = model.predict(future[['YEAR']])
            st.line_chart(pd.concat([yearly, future]), x='YEAR', y='TOTAL_CLAIM_COST')

except Exception as e:
    st.error("A runtime error occurred.")
    st.exception(e)
    
