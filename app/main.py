import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.linear_model import LinearRegression

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Health Equity OS", layout="wide")

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
    
    # NEW: Average Insurance Coverage % (Coverage / Expenses)
    p['INSURANCE_COVERAGE_PCT'] = (p['HEALTHCARE_COVERAGE'] / (p['HEALTHCARE_EXPENSES'] + 1) * 100).clip(0, 100)
    p['INCOME_TIER'] = p['INCOME'].apply(lambda x: 'Low' if x < 35000 else ('Middle' if x < 85000 else 'High'))
    
    # Load Encounters
    e_files = list(data_dir.glob("encounters*.csv"))
    e = pd.concat([pd.read_csv(f) for f in e_files if f.stat().st_size > 0], ignore_index=True)
    e['START'] = pd.to_datetime(e['START'])
    e['YEAR'] = e['START'].dt.year
    
    return pd.merge(e, p, left_on='PATIENT', right_on='Id', how='inner')

# --- 3. HELPER: PREDICTIVE MODEL ---
def run_projection(df, dep_var):
    yearly = df.groupby('YEAR')[dep_var].mean().reset_index()
    X = yearly[['YEAR']].values
    y = yearly[dep_var].values
    
    model = LinearRegression().fit(X, y)
    future_years = np.array(range(yearly['YEAR'].max() + 1, 2031)).reshape(-1, 1)
    future_preds = model.predict(future_years)
    
    hist = pd.DataFrame({'Year': yearly['YEAR'], 'Value': y, 'Status': 'Actual'})
    proj = pd.DataFrame({'Year': future_years.flatten(), 'Value': future_preds, 'Status': 'Projected'})
    return pd.concat([hist, proj])

# --- 4. MAIN NAVIGATION ---
try:
    df = load_and_prep_data()
    
    st.sidebar.title("🗂️ Navigation")
    page = st.sidebar.radio("Go to:", ["Welcome", "Interactive Map", "Comparative Analysis", "Predictive Forecasting"])

    # --- TAB: WELCOME ---
    if page == "Welcome":
        st.title("🏥 Health Equity Insights Platform")
        st.markdown("""
        ### Strategic Overview
        This dashboard identifies **Vertical Equity Gaps**—disparities in healthcare access based on wealth and demographics.
        
        **What's New:**
        * **Interactive Maps:** Analyze California county-level burdens.
        * **Intersectionality:** Compare how gender, race, and income collide.
        * **Forecasting:** See where healthcare costs are projected to be by 2030.
        """)

    # --- TAB 1: INTERACTIVE CHOROPLETH ---
    elif page == "Interactive Map":
        st.title("🗺️ California Regional Analysis")
        st.info("💡 **Quick Summary:** Use this map to see which counties face the highest healthcare burdens. Darker colors indicate higher average costs. Hover for income and insurance details.")
        
        map_stats = df.groupby('COUNTY').agg({
            'TOTAL_CLAIM_COST': 'mean',
            'INCOME': 'mean',
            'INSURANCE_COVERAGE_PCT': 'mean'
        }).reset_index()

        fig = px.choropleth(map_stats, locations='COUNTY', locationmode='USA-states', 
                            color='TOTAL_CLAIM_COST', scope="usa",
                            hover_data={'INCOME': ':,.0f', 'INSURANCE_COVERAGE_PCT': ':.1f%'},
                            color_continuous_scale="Viridis", title="Avg Claims Cost by County")
        fig.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("🔍 Deep-Dive: County Statistics")
        selected_county = st.selectbox("Select a County to generate specific graphs:", sorted(df['COUNTY'].unique()))
        c_df = df[df['COUNTY'] == selected_county]
        
        col1, col2 = st.columns(2)
        with col1:
            dep_var = st.selectbox("Dependent Variable:", ['TOTAL_CLAIM_COST', 'INCOME', 'INSURANCE_COVERAGE_PCT'])
            indep_var = st.selectbox("Independent Variable (Demographic):", ['INCOME_TIER', 'GENDER', 'RACE'])
            
            c_chart = px.bar(c_df.groupby(indep_var)[dep_var].mean().reset_index(), 
                             x=indep_var, y=dep_var, color=indep_var, title=f"{selected_county}: {dep_var} by {indep_var}")
            st.plotly_chart(c_chart, use_container_width=True)

    # --- TAB 2: COMPARATIVE ANALYSIS ---
    elif page == "Comparative Analysis":
        st.title("⚖️ Population Comparison")
        st.info("💡 **Quick Summary:** This section allows you to compare different populations side-by-side to identify equity gaps in cost and coverage.")
        
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

    # --- TAB 4: PREDICTIVE TOOL ---
    elif page == "Predictive Forecasting":
        st.title("🔮 Predictive Analytics Tool")
        st.info("💡 **Quick Summary:** This graph shows historical trends (Past to Present) and projects where healthcare rates are headed through 2030 using Linear Regression.")
        
        target = st.selectbox("Select Variable to Project into the Future:", ['TOTAL_CLAIM_COST', 'INCOME', 'INSURANCE_COVERAGE_PCT'])
        
        proj_data = run_projection(df, target)
        
        fig = px.line(proj_data, x='Year', y='Value', color='Status', markers=True, 
                     title=f"2030 Projection: {target.replace('_', ' ')}")
        st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"**Insight:** By 2030, the projected {target.lower()} is expected to reach **${proj_data.iloc[-1]['Value']:,.2f}**.")

except Exception as e:
    st.error("🚨 System Update in Progress. Please ensure data files are valid.")
    st.exception(e)
