import streamlit as st
import pandas as pd
import altair as alt
import os
from pathlib import Path

# --- 1. SETTINGS & STYLING ---
st.set_page_config(page_title="Health Equity OS", layout="wide", page_icon="🏥")

st.markdown("""
    <style>
    .big-header {font-size: 50px !important; color: #1E3A8A; font-weight: 800; text-align: center; margin-bottom: 30px; border-bottom: 5px solid #FF4B4B;}
    .sub-header {background-color: #1E3A8A; color: white; padding: 12px; border-radius: 8px; font-size: 20px; margin-top: 20px;}
    .card {background-color: #f8f9fa; border: 1px solid #dee2e6; padding: 20px; border-radius: 10px; height: 100%;}
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINE ---
@st.cache_data
def get_all_data():
    root_path = Path(__file__).resolve().parent.parent
    data_dir = root_path / "data"
    
    p = pd.read_csv(data_dir / "patients.csv")
    p['BIRTHDATE'] = pd.to_datetime(p['BIRTHDATE'])
    p['AGE'] = (pd.Timestamp.today() - p['BIRTHDATE']).dt.days // 365
    for c in ['INCOME', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']:
        p[c] = pd.to_numeric(p[c].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce').fillna(0)
    
    p['INCOME_TIER'] = p['INCOME'].apply(lambda x: 'Low' if x < 35000 else ('Middle' if x < 85000 else 'High'))
    p['INSURANCE_STATUS'] = p['HEALTHCARE_COVERAGE'].apply(
        lambda x: 'Under-Insured' if x < 5000 else ('Standard' if x < 15000 else 'Premium')
    )
    
    e = pd.read_csv(data_dir / "encounters_part_1.csv")
    df = pd.merge(e, p, left_on='PATIENT', right_on='Id', how='inner')
    return df

def get_interpretation(data, metric, group_col):
    if data.empty: return "No data available for selected filters."
    stats = data.groupby(group_col)[metric].mean().sort_values(ascending=False)
    top_group = stats.index[0]
    top_val = stats.iloc[0]
    return f"💡 **Observation:** The **{top_group}** group is currently seeing the highest average cost of **${top_val:,.2f}**, indicating a key priority for equity intervention."

# --- 3. PAGE FUNCTIONS ---

def show_overview():
    st.markdown('<p class="big-header">HEALTH EQUITY INSIGHTS PLATFORM</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color: #eef2ff; padding: 25px; border-radius: 15px; border-left: 8px solid #1E3A8A; margin-bottom: 30px;">
    <h3>Project Mission</h3>
    This application identifies <strong>Vertical Equity Gaps</strong> in California's healthcare landscape. 
    By intersecting clinical data with socio-economic indicators, we empower stakeholders to make informed resource allocations.
    <br><br><strong><a href="https://github.com/Jagadeeshwari1/Health_care" target="_blank">🔗 Detailed Project Documentation</a></strong>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card"><p class="sub-header">🎯 Decision Makers</p>'
                    '<ul><li><b>Public Health Officials:</b> Resource distribution</li>'
                    '<li><b>Policy Analysts:</b> Equity legislation</li></ul></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><p class="sub-header">👥 End Users</p>'
                    '<ul><li><b>Data Analysts:</b> Clinical trends</li>'
                    '<li><b>Social Workers:</b> Patient advocacy</li></ul></div>', unsafe_allow_html=True)

def show_income_page(data):
    st.title("💰 Income Analysis")
    st.sidebar.subheader("Income Filters")
    tier_filter = st.sidebar.multiselect("Select Tiers", sorted(data['INCOME_TIER'].unique()), default=data['INCOME_TIER'].unique())
    filtered = data[data['INCOME_TIER'].isin(tier_filter)]
    
    chart = alt.Chart(filtered).mark_bar().encode(
        x=alt.X('INCOME_TIER', sort=['Low', 'Middle', 'High']),
        y='mean(TOTAL_CLAIM_COST)',
        color=alt.Color('INCOME_TIER', scale=alt.Scale(
            domain=['High', 'Middle', 'Low'],
            range=['#FF4B4B', '#FFD700', '#2E8B57'] 
        ), legend=None)
    ).properties(height=450)
    st.altair_chart(chart, use_container_width=True)
    st.info(get_interpretation(filtered, 'TOTAL_CLAIM_COST', 'INCOME_TIER'))

def show_age_page(data):
    st.title("📅 Age Analysis")
    st.sidebar.subheader("Age Controls")
    age_range = st.sidebar.slider("Filter Age Range", 0, 100, (0, 100))
    filtered = data[(data['AGE'] >= age_range[0]) & (data['AGE'] <= age_range[1])]
    
    age_trend = filtered.groupby('AGE')['TOTAL_CLAIM_COST'].mean().reset_index()
    chart = alt.Chart(age_trend).mark_line(color='#1E3A8A', strokeWidth=3).encode(
        x='AGE', y='TOTAL_CLAIM_COST'
    ).properties(height=500)
    st.altair_chart(chart, use_container_width=True)
    st.info(get_interpretation(filtered, 'TOTAL_CLAIM_COST', 'AGE'))

def show_health_conditions(data):
    st.title("🩺 Health Conditions")
    st.sidebar.subheader("Income Filter")
    selected_income = st.sidebar.selectbox("Filter by Income Group", sorted(data['INCOME_TIER'].unique()))
    
    filtered = data[data['INCOME_TIER'] == selected_income]
    top_problems = filtered.groupby('DESCRIPTION')['TOTAL_CLAIM_COST'].mean().sort_values(ascending=False).head(10).reset_index()
    
    chart = alt.Chart(top_problems).mark_bar().encode(
        x=alt.X('TOTAL_CLAIM_COST', title="Avg Cost ($)"),
        y=alt.Y('DESCRIPTION', sort='-x', title=None),
        color=alt.Color('DESCRIPTION', scale=alt.Scale(scheme='tableau20'), legend=None)
    ).properties(height=500)
    st.altair_chart(chart, use_container_width=True)
    # ADDED INTERPRETATION
    st.info(get_interpretation(filtered, 'TOTAL_CLAIM_COST', 'DESCRIPTION'))

def show_geography_page(data):
    st.title("🌍 Geography Analysis")
    st.sidebar.subheader("County Filters")
    all_counties = sorted(data['COUNTY'].unique())
    selected_counties = st.sidebar.multiselect("Select Counties", all_counties, default=all_counties[:10])
    
    filtered = data[data['COUNTY'].isin(selected_counties)]
    
    st.subheader("Top 10 High-Cost Counties")
    county_stats = filtered.groupby('COUNTY')['TOTAL_CLAIM_COST'].mean().sort_values(ascending=False).head(10).reset_index()
    
    chart = alt.Chart(county_stats).mark_bar(color='#4B0082').encode(
        x=alt.X('TOTAL_CLAIM_COST', title="Average Claim Cost ($)"),
        y=alt.Y('COUNTY', sort='-x', title="County Name")
    ).properties(height=500)
    st.altair_chart(chart, use_container_width=True)
    # ADDED INTERPRETATION
    st.info(get_interpretation(county_stats, 'TOTAL_CLAIM_COST', 'COUNTY'))

def show_insurance_page(data):
    st.title("🛡️ Insurance Coverage")
    st.sidebar.subheader("Coverage Filter")
    status_filter = st.sidebar.multiselect("Select Status", data['INSURANCE_STATUS'].unique(), default=data['INSURANCE_STATUS'].unique())
    filtered = data[data['INSURANCE_STATUS'].isin(status_filter)]
    
    chart = alt.Chart(filtered).mark_bar().encode(
        x='INSURANCE_STATUS', 
        y=alt.Y('mean(HEALTHCARE_EXPENSES)', title="Average Expenses ($)"),
        color=alt.Color('INSURANCE_STATUS', scale=alt.Scale(scheme='set2'))
    ).properties(height=450)
    st.altair_chart(chart, use_container_width=True)
    # ADDED INTERPRETATION
    st.info(get_interpretation(filtered, 'HEALTHCARE_EXPENSES', 'INSURANCE_STATUS'))

def show_ledger(data):
    st.title("📑 Data Ledger")
    search = st.text_input("Search records...", "").lower()
    cols = ['PATIENT', 'CITY', 'COUNTY', 'AGE', 'INCOME_TIER', 'INSURANCE_STATUS', 'DESCRIPTION', 'TOTAL_CLAIM_COST']
    existing = [c for c in cols if c in data.columns]
    
    if search:
        data = data[data.apply(lambda r: r.astype(str).str.contains(search, case=False).any(), axis=1)]
    st.dataframe(data[existing].head(500), use_container_width=True)

# --- 4. MAIN ROUTING ---
try:
    full_df = get_all_data()
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Overview", "Income Analysis", "Age Analysis", "Health Conditions", "Geography", "Insurance Coverage", "Ledger"])

    if page == "Overview": show_overview()
    elif page == "Income Analysis": show_income_page(full_df)
    elif page == "Age Analysis": show_age_page(full_df)
    elif page == "Health Conditions": show_health_conditions(full_df)
    elif page == "Geography": show_geography_page(full_df)
    elif page == "Insurance Coverage": show_insurance_page(full_df)
    elif page == "Ledger": show_ledger(full_df)

except Exception as e:
    st.error("System Configuration Error")
    st.exception(e)
