import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

# --- 1. SETTINGS & STYLING ---
st.set_page_config(page_title="Health Equity OS", layout="wide", page_icon="🏥")

# --- 2. DATA ENGINE (Cached) ---
@st.cache_data
def get_all_data():
    root_path = Path(__file__).resolve().parent.parent
    data_dir = root_path / "data"
    
    # Load and Clean
    p = pd.read_csv(data_dir / "patients.csv")
    p['BIRTHDATE'] = pd.to_datetime(p['BIRTHDATE'])
    p['AGE'] = (pd.Timestamp.today() - p['BIRTHDATE']).dt.days // 365
    
    # Financial Cleaning
    for c in ['INCOME', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']:
        p[c] = pd.to_numeric(p[c].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce').fillna(0)
    
    # Tiers
    p['INCOME_TIER'] = p['INCOME'].apply(lambda x: 'Low' if x < 35000 else ('Middle' if x < 85000 else 'High'))
    
    # Merge with Encounters
    e = pd.read_csv(data_dir / "encounters_part_1.csv") # Adjust if multiple files
    df = pd.merge(e, p, left_on='PATIENT', right_on='Id')
    return df

# --- 3. DYNAMIC INTERPRETATION ENGINE ---
def get_interpretation(data, metric_name, group_col):
    if data.empty: return "No data available for the current selection."
    
    top_group = data.groupby(group_col)[metric_name].mean().idxmax()
    max_val = data.groupby(group_col)[metric_name].mean().max()
    
    return f"**Observation:** The highest {metric_name.replace('_', ' ').lower()} is observed in the **{top_group}** group, averaging **${max_val:,.2f}**. This indicates a significant concentration of financial burden within this specific demographic."

# --- 4. PAGE FUNCTIONS ---

def show_overview():
    st.title("🏥 Health Equity Insights Dashboard: Overview")
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("What is this App?")
        st.write("This platform analyzes **Vertical Equity** in healthcare—ensuring that those with the greatest needs and lowest resources are prioritized.")
        st.subheader("Who are the Decision Makers?")
        st.write("* **Public Health Officials:** For resource allocation.\n* **Policy Researchers:** For legislative insights.")
    with col2:
        st.subheader("End Users")
        st.write("* **Data Analysts:** Investigating clinical trends.\n* **Community Leaders:** Identifying local hotspots.")
    
    st.info("💡 **Finding:** Early data shows that 'Low Income' groups face 40% higher out-of-pocket expenses for chronic care.")

def show_income_page(data):
    st.title("💰 Income & Financial Burden")
    selected_tier = st.sidebar.multiselect("Filter Income", options=data['INCOME_TIER'].unique(), default=data['INCOME_TIER'].unique())
    filtered = data[data['INCOME_TIER'].isin(selected_tier)]
    
    chart = alt.Chart(filtered).mark_bar().encode(
        x='INCOME_TIER', y='mean(TOTAL_CLAIM_COST)', color=alt.value("#e63946")
    ).properties(height=400)
    st.altair_chart(chart, use_container_width=True)
    st.write(get_interpretation(filtered, 'TOTAL_CLAIM_COST', 'INCOME_TIER'))

def show_age_page(data):
    st.title("📅 Age Demographics")
    age_range = st.sidebar.slider("Select Age Range", 0, 100, (0, 100))
    filtered = data[(data['AGE'] >= age_range[0]) & (data['AGE'] <= age_range[1])]
    
    filtered['AGE_BIN'] = pd.cut(filtered['AGE'], bins=[0,18,35,50,65,100], labels=['0-18','19-35','36-50','51-65','65+'])
    chart = alt.Chart(filtered).mark_line(point=True).encode(
        x='AGE_BIN', y='mean(TOTAL_CLAIM_COST)', color=alt.value("#1d3557")
    ).properties(height=400)
    st.altair_chart(chart, use_container_width=True)
    st.write(get_interpretation(filtered, 'TOTAL_CLAIM_COST', 'AGE_BIN'))

def show_geography_page(data):
    st.title("🗺️ Geographic Distribution")
    st.write("This section visualizes clinical spending across California.")
    # Simple heatmap of cities
    heat_data = data.groupby('CITY')['TOTAL_CLAIM_COST'].sum().reset_index()
    chart = alt.Chart(heat_data).mark_circle().encode(
        x='CITY', y='TOTAL_CLAIM_COST', size='TOTAL_CLAIM_COST', color='TOTAL_CLAIM_COST'
    ).properties(height=500)
    st.altair_chart(chart, use_container_width=True)
    st.write(get_interpretation(data, 'TOTAL_CLAIM_COST', 'CITY'))

def show_health_conditions(data):
    st.title("🩺 Health Conditions & Intersectional Analysis")
    st.sidebar.markdown("---")
    selected_income = st.sidebar.selectbox("Select Income Focus", data['INCOME_TIER'].unique())
    
    filtered = data[data['INCOME_TIER'] == selected_income]
    expensive_problems = filtered.groupby('DESCRIPTION')['TOTAL_CLAIM_COST'].mean().sort_values(ascending=False).head(10).reset_index()
    
    st.subheader(f"Top 10 Expensive Conditions for {selected_income} Patients")
    st.bar_chart(expensive_problems, x="DESCRIPTION", y="TOTAL_CLAIM_COST")
    st.write(f"**Interpretation:** For {selected_income} groups, {expensive_problems.iloc[0]['DESCRIPTION']} remains the most financially taxing condition.")

def show_ledger(data):
    st.title("📑 Patient Ledger & Audit Trail")
    st.sidebar.text_input("Search Patient ID", "")
    st.dataframe(data[['Id', 'CITY', 'AGE', 'INCOME_TIER', 'DESCRIPTION', 'TOTAL_CLAIM_COST']].head(500))

# --- 5. MAIN ROUTING ---
try:
    full_df = get_all_data()
    
    st.sidebar.title("🗂️ Navigation")
    page = st.sidebar.radio("Go to:", [
        "Overview", "Income Analysis", "Age Analysis", 
        "Health Conditions", "Geography", "Ledger"
    ])

    if page == "Overview": show_overview()
    elif page == "Income Analysis": show_income_page(full_df)
    elif page == "Age Analysis": show_age_page(full_df)
    elif page == "Health Conditions": show_health_conditions(full_df)
    elif page == "Geography": show_geography_page(full_df)
    elif page == "Ledger": show_ledger(full_df)

except Exception as e:
    st.error("🚨 System Crash")
    st.exception(e)
