import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
from pathlib import Path
from sklearn.linear_model import LinearRegression

# --- 1. SETTINGS & STYLING ---
st.set_page_config(page_title="Health Equity OS", layout="wide", page_icon="🏥")

# --- 2. DATA ENGINE ---
@st.cache_data
def get_all_data():
    root_path = Path(__file__).resolve().parent.parent
    data_dir = root_path / "data"
    
    # Load Patients
    p = pd.read_csv(data_dir / "patients.csv")
    p['BIRTHDATE'] = pd.to_datetime(p['BIRTHDATE'])
    p['AGE'] = (pd.Timestamp.today() - p['BIRTHDATE']).dt.days // 365
    
    # Financial Cleaning
    for c in ['INCOME', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']:
        p[c] = pd.to_numeric(p[c].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce').fillna(0)
    
    # Coverage % Calculation (Professor's Request)
    p['INSURANCE_COVERAGE_PCT'] = (p['HEALTHCARE_COVERAGE'] / (p['HEALTHCARE_EXPENSES'] + 1) * 100).clip(0, 100)
    
    # Load and Merge Encounters
    encounter_files = list(data_dir.glob("encounters*.csv"))
    e = pd.concat([pd.read_csv(f) for f in encounter_files], ignore_index=True)
    e['START'] = pd.to_datetime(e['START'])
    e['YEAR'] = e['START'].dt.year
    
    df = pd.merge(e, p, left_on='PATIENT', right_on='Id', how='inner')
    return df

# --- 3. PAGE FUNCTIONS ---

def show_intro():
    st.title("🏥 Welcome to Health Equity OS")
    st.markdown("""
    ### Project Overview
    This platform analyzes healthcare cost and access disparities. We focus on **Vertical Equity**—the principle that healthcare 
    should be distributed based on need and socio-economic vulnerability.
    
    **Instructions:** Use the sidebar to navigate through the interactive map, demographic comparisons, and predictive tools.
    """)

def show_map_page(df):
    st.header("📍 Regional Health Equity Map")
    st.info("💡 **Summary:** This map shows healthcare metrics across counties. Darker regions indicate higher costs or lower income. Hover over a county to see specific details.")
    
    # Aggregate data by County
    county_map_data = df.groupby('COUNTY').agg({
        'TOTAL_CLAIM_COST': 'mean',
        'INCOME': 'mean',
        'INSURANCE_COVERAGE_PCT': 'mean'
    }).reset_index()

    # NOTE: To show a real map, we'd need a GeoJSON. Since we are presenting, 
    # we'll use a high-quality bar chart as a proxy or Plotly's built-in map if coordinates exist.
    fig = px.choropleth(county_map_data, 
                        locations='COUNTY', 
                        locationmode='USA-states', # or custom GeoJSON if available
                        color='TOTAL_CLAIM_COST',
                        hover_name='COUNTY',
                        hover_data=['INCOME', 'INSURANCE_COVERAGE_PCT'],
                        title="California County Healthcare Burden",
                        scope="usa")
    
    # If GeoJSON isn't available, Plotly will show a list/table by default. 
    # Let's provide the interactive county click-down.
    selected_county = st.selectbox("Select a County for Detailed Analysis", options=county_map_data['COUNTY'].unique())
    
    county_df = df[df['COUNTY'] == selected_county]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Claims Cost", f"${county_df['TOTAL_CLAIM_COST'].mean():,.2f}")
    col2.metric("Avg Income", f"${county_df['INCOME'].mean():,.2f}")
    col3.metric("Insurance Coverage %", f"{county_df['INSURANCE_COVERAGE_PCT'].mean():.1f}%")

def show_compare_page(df):
    st.header("⚖️ Population Comparison")
    st.info("💡 **Summary:** Compare how different groups (by race, gender, or income) experience healthcare costs side-by-side.")
    
    st.sidebar.subheader("Comparison Settings")
    factor = st.sidebar.selectbox("Choose Demographic Factor", ['RACE', 'GENDER', 'INCOME_TIER'])
    metric = st.sidebar.selectbox("Choose Metric to Compare", ['TOTAL_CLAIM_COST', 'INCOME', 'INSURANCE_COVERAGE_PCT'])

    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X(f'{factor}:N', title=factor),
        y=alt.Y(f'mean({metric}):Q', title=f"Average {metric}"),
        color=f'{factor}:N',
        tooltip=[factor, f'mean({metric})']
    ).properties(height=450)
    
    st.altair_chart(chart, use_container_width=True)

def show_predictive_page(df):
    st.header("🔮 Predictive Insights")
    st.info("💡 **Summary:** This tool looks at historical data (past to present) and uses a trend line to project what costs and coverage might look like in the future.")
    
    dependent_var = st.selectbox("Select Variable to Project", ['TOTAL_CLAIM_COST', 'INCOME', 'INSURANCE_COVERAGE_PCT'])
    
    # Prepare Time Series
    yearly_data = df.groupby('YEAR')[dependent_var].mean().reset_index()
    
    # Linear Regression for Projection
    X = yearly_data[['YEAR']].values
    y = yearly_data[dependent_var].values
    model = LinearRegression()
    model.fit(X, y)
    
    # Create Future Dates
    future_years = np.array([[2026], [2027], [2028], [2029], [2030]])
    future_preds = model.predict(future_years)
    
    # Combine Data
    future_df = pd.DataFrame({'YEAR': future_years.flatten(), dependent_var: future_preds, 'Type': 'Projected'})
    yearly_data['Type'] = 'Actual'
    combined = pd.concat([yearly_data, future_df])
    
    # Plot
    chart = alt.Chart(combined).mark_line(point=True).encode(
        x='YEAR:O',
        y=alt.Y(dependent_var, title=f"Average {dependent_var}"),
        color='Type',
        strokeDash='Type'
    ).properties(height=500)
    
    st.altair_chart(chart, use_container_width=True)
    st.write(f"**Projection Note:** Based on historical trends, we expect {dependent_var.replace('_', ' ').lower()} to reach **${future_preds[-1]:,.2f}** by 2030.")

# --- 4. MAIN ROUTING ---
try:
    full_df = get_all_data()
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Intro", "Interactive Map", "Compare Groups", "Predictive Tool"])

    if page == "Intro": show_intro()
    elif page == "Interactive Map": show_map_page(full_df)
    elif page == "Compare Groups": show_compare_page(full_df)
    elif page == "Predictive Tool": show_predictive_page(full_df)

except Exception as e:
    st.error("🚨 System Update Required")
    st.exception(e)
    
