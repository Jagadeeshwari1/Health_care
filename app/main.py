import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

# --- 1. SETTINGS & STYLING ---
st.set_page_config(page_title="Health Equity OS", layout="wide", page_icon="🏥")

# Custom CSS for Colorful Headers
st.markdown("""
    <style>
    .main-header {font-size:35px !important; color: #1E3A8A; font-weight: bold; border-bottom: 3px solid #1E3A8A; margin-bottom: 20px;}
    .sub-header {background-color: #1E3A8A; color: white; padding: 10px; border-radius: 5px; margin-top: 20px;}
    .highlight-blue {background-color: #DBEAFE; padding: 15px; border-left: 5px solid #2563EB; border-radius: 5px; margin-bottom: 10px;}
    .highlight-green {background-color: #DCFCE7; padding: 15px; border-left: 5px solid #16A34A; border-radius: 5px; margin-bottom: 10px;}
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
    
    e = pd.read_csv(data_dir / "encounters_part_1.csv")
    df = pd.merge(e, p, left_on='PATIENT', right_on='Id', how='inner')
    return df

# --- 3. DYNAMIC INTERPRETATION ENGINE ---
def get_interpretation(data, metric, group_col):
    if data.empty: return "No data available."
    stats = data.groupby(group_col)[metric].mean().sort_values(ascending=False)
    top_group = stats.index[0]
    top_val = stats.iloc[0]
    return f"💡 **Observation:** The **{top_group}** group is currently seeing the highest average cost of **${top_val:,.2f}**, suggesting a focal point for policy intervention."

# --- 4. PAGE FUNCTIONS ---

def show_overview():
    st.markdown('<p class="main-header">🏥 Health Equity Insights Platform</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight-blue">
    <strong>Project Mission:</strong> This platform is designed to identify <strong>Vertical Equity gaps</strong>. 
    By analyzing socio-economic status against clinical outcomes, we enable data-driven resource allocation.
    <br><a href="https://github.com/Jagadeeshwari1/Health_care" target="_blank">View Detailed Research Documentation</a>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="sub-header">🎯 Key Decision Makers</p>', unsafe_allow_html=True)
        st.write("1. **Public Health Officials:** Budget and resource allocation.")
        st.write("2. **Insurance Underwriters:** Identifying high-risk cost clusters.")
        st.write("3. **Policy Legislators:** Drafting equity-focused healthcare laws.")
    
    with col2:
        st.markdown('<p class="sub-header">👥 Primary End Users</p>', unsafe_allow_html=True)
        st.write("1. **Clinical Analysts:** Monitoring patient outcome trends.")
        st.write("2. **Social Workers:** Connecting at-risk groups to coverage.")
        st.write("3. **Community Leaders:** Advocating for local health needs.")

def show_income_page(data):
    st.title("💰 Income & Financial Burden")
    # High-Red, Middle-Yellow, Low-Green
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('INCOME_TIER', sort=['Low', 'Middle', 'High']),
        y=alt.Y('mean(TOTAL_CLAIM_COST)', title="Avg Claim Cost ($)"),
        color=alt.Color('INCOME_TIER', scale=alt.Scale(
            domain=['High', 'Middle', 'Low'],
            range=['#FF4B4B', '#FFD700', '#2E8B57']
        ), legend=None)
    ).properties(height=450)
    st.altair_chart(chart, use_container_width=True)
    st.info(get_interpretation(data, 'TOTAL_CLAIM_COST', 'INCOME_TIER'))

def show_age_page(data):
    st.title("📅 Age-Based Expense Analysis")
    age_range = st.sidebar.slider("Select Age Window", 0, 100, (0, 100))
    filtered = data[(data['AGE'] >= age_range[0]) & (data['AGE'] <= age_range[1])].copy()
    filtered['AGE_GROUP'] = pd.cut(filtered['AGE'], bins=[0,18,35,50,65,120], labels=['0-18','19-35','36-50','51-65','65+'])
    
    chart = alt.Chart(filtered).mark_area(opacity=0.6, color="#1F77B4").encode(
        x='AGE_GROUP', y='mean(TOTAL_CLAIM_COST)'
    ).properties(height=400)
    st.altair_chart(chart, use_container_width=True)
    st.info(get_interpretation(filtered, 'TOTAL_CLAIM_COST', 'AGE_GROUP'))

def show_health_conditions(data):
    st.title("🩺 Intersectional Health Analysis")
    top_problems = data.groupby('DESCRIPTION')['TOTAL_CLAIM_COST'].mean().sort_values(ascending=False).head(10).reset_index()
    
    chart = alt.Chart(top_problems).mark_bar().encode(
        x=alt.X('TOTAL_CLAIM_COST', title="Average Cost ($)"),
        y=alt.Y('DESCRIPTION', sort='-x', title=None),
        color=alt.Color('DESCRIPTION', scale=alt.Scale(scheme='tableau10'), legend=None)
    ).properties(height=500)
    st.altair_chart(chart, use_container_width=True)
    st.info(get_interpretation(data, 'TOTAL_CLAIM_COST', 'DESCRIPTION'))

def show_geography_page(data):
    st.title("🌍 Geographic Intensity")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Cities by Expense")
        city_stats = data.groupby('CITY')['TOTAL_CLAIM_COST'].mean().sort_values(ascending=False).head(10).reset_index()
        st.bar_chart(city_stats, x='CITY', y='TOTAL_CLAIM_COST', color="#4B0082")
    with col2:
        st.subheader("County Distribution")
        county_stats = data.groupby('COUNTY')['TOTAL_CLAIM_COST'].sum().reset_index()
        st.vega_lite_chart(county_stats, {
            'mark': {'type': 'arc', 'innerRadius': 40},
            'encoding': {
                'theta': {'field': 'TOTAL_CLAIM_COST', 'type': 'quantitative'},
                'color': {'field': 'COUNTY', 'type': 'nominal', 'scale': {'scheme': 'category20b'}}
            }
        }, use_container_width=True)
    st.info(get_interpretation(data, 'TOTAL_CLAIM_COST', 'CITY'))

def show_ledger(data):
    st.title("📑 Data Audit Ledger")
    search = st.text_input("🔍 Search by Patient ID, City, or Condition", "").lower()
    
    display_cols = ['PATIENT', 'CITY', 'AGE', 'INCOME_TIER', 'DESCRIPTION', 'TOTAL_CLAIM_COST']
    existing = [c for c in display_cols if c in data.columns]
    
    if search:
        # Filter across all visible columns
        filtered_df = data[data.apply(lambda row: row.astype(str).str.contains(search, case=False).any(), axis=1)]
    else:
        filtered_df = data
        
    st.dataframe(filtered_df[existing].head(500), use_container_width=True)

# --- 5. MAIN ROUTING ---
try:
    full_df = get_all_data()
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select View", ["Overview", "Income Analysis", "Age Analysis", "Health Conditions", "Geography", "Ledger"])

    if page == "Overview": show_overview()
    elif page == "Income Analysis": show_income_page(full_df)
    elif page == "Age Analysis": show_age_page(full_df)
    elif page == "Health Conditions": show_health_conditions(full_df)
    elif page == "Geography": show_geography_page(full_df)
    elif page == "Ledger": show_ledger(full_df)

except Exception as e:
    st.error("🚨 System Configuration Error")
    st.exception(e)
