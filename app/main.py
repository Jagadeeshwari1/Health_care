import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

# --- 1. SETTINGS & STYLING ---
st.set_page_config(page_title="Intersectionality & Health Equity", layout="wide", page_icon="🏥")

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
    
    # Load and Clean Patients
    p = pd.read_csv(data_dir / "patients.csv")
    p['BIRTHDATE'] = pd.to_datetime(p['BIRTHDATE'])
    p['AGE'] = (pd.Timestamp.today() - p['BIRTHDATE']).dt.days // 365
    
    # Financial Cleaning
    for c in ['INCOME', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']:
        p[c] = pd.to_numeric(p[c].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce').fillna(0)
    
    # Feature Engineering
    p['INCOME_TIER'] = p['INCOME'].apply(lambda x: 'Low' if x < 35000 else ('Middle' if x < 85000 else 'High'))
    p['INSURANCE_STATUS'] = p['HEALTHCARE_COVERAGE'].apply(
        lambda x: 'Under-Insured' if x < 5000 else ('Standard' if x < 15000 else 'Premium')
    )
    
    # Load and Merge Encounters
    e = pd.read_csv(data_dir / "encounters_part_1.csv")
    df = pd.merge(e, p, left_on='PATIENT', right_on='Id', how='inner')
    return df

# --- 3. DYNAMIC INTERPRETATION ENGINE ---
def get_interpretation(data, metric, group_col):
    if data.empty: return "No data available for the current intersection of filters."
    stats = data.groupby(group_col)[metric].mean().sort_values(ascending=False)
    top_group = stats.index[0]
    top_val = stats.iloc[0]
    return f"💡 **Intersectional Insight:** Within this specific cohort, **{top_group}** individuals face the highest burden (Avg: **${top_val:,.2f}**)."

# --- 4. GLOBAL SIDEBAR FILTERS (FOR INTERSECTIONALITY) ---
def sidebar_filters(df):
    st.sidebar.title("🎛️ Intersectional Filters")
    st.sidebar.info("Filters apply across all analytical tabs.")
    
    # 1. Age Range
    age_range = st.sidebar.slider("Age Bracket", 0, 100, (0, 100))
    
    # 2. Gender (Re-added)
    genders = sorted(df['GENDER'].unique()) if 'GENDER' in df.columns else []
    sel_gender = st.sidebar.multiselect("Gender Identity", genders, default=genders)
    
    # 3. Race (Re-added)
    races = sorted(df['RACE'].unique()) if 'RACE' in df.columns else []
    sel_race = st.sidebar.multiselect("Race/Ethnicity", races, default=races)
    
    # 4. Wealth/Income
    tiers = ['Low', 'Middle', 'High']
    sel_income = st.sidebar.multiselect("Income Tier", tiers, default=tiers)
    
    # Apply Filtering Logic
    mask = (
        (df['AGE'] >= age_range[0]) & (df['AGE'] <= age_range[1]) &
        (df['GENDER'].isin(sel_gender)) &
        (df['RACE'].isin(sel_race)) &
        (df['INCOME_TIER'].isin(sel_income))
    )
    return df[mask]

# --- 5. PAGE FUNCTIONS ---

def show_overview():
    st.markdown('<p class="big-header">HEALTH EQUITY INSIGHTS PLATFORM</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color: #eef2ff; padding: 25px; border-radius: 15px; border-left: 8px solid #1E3A8A; margin-bottom: 30px;">
    <h3>Project Mission</h3>
    This application identifies <strong>Vertical Equity Gaps</strong> across the healthcare landscape. 
    By intersecting clinical data with multi-dimensional socio-economic indicators—including age, race, gender, and wealth—we empower stakeholders to make informed, intersectional resource allocations.
    <br><br><strong><a href="https://github.com/Jagadeeshwari1/Health_care" target="_blank">🔗 Detailed Project Documentation</a></strong>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card"><p class="sub-header">🎯 Decision Makers</p>'
                    '<ul><li><b>Public Health Officials:</b> Global resource distribution</li>'
                    '<li><b>Policy Analysts:</b> Intersectional equity legislation</li></ul></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><p class="sub-header">👥 End Users</p>'
                    '<ul><li><b>Data Analysts:</b> Multi-factor clinical trends</li>'
                    '<li><b>Advocacy Groups:</b> Targeted community support</li></ul></div>', unsafe_allow_html=True)

def show_multivariate_analysis(filtered_df):
    st.title("🧩 Intersectional Analysis")
    st.markdown("Analyzing the cumulative impact of Age, Gender, and Income on Claim Costs.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cost by Gender & Income Tier")
        chart = alt.Chart(filtered_df).mark_bar().encode(
            x='GENDER:N',
            y='mean(TOTAL_CLAIM_COST):Q',
            color=alt.Color('INCOME_TIER', scale=alt.Scale(domain=['High', 'Middle', 'Low'], range=['#FF4B4B', '#FFD700', '#2E8B57'])),
            column='INCOME_TIER:N'
        ).properties(width=150, height=400)
        st.altair_chart(chart)
    
    with col2:
        st.subheader("Cost Distribution by Race")
        chart_race = alt.Chart(filtered_df).mark_bar(color='#4B0082').encode(
            x=alt.X('mean(TOTAL_CLAIM_COST):Q', title="Avg Claim Cost ($)"),
            y=alt.Y('RACE:N', sort='-x', title=None)
        ).properties(height=400)
        st.altair_chart(chart_race, use_container_width=True)
    
    st.info(get_interpretation(filtered_df, 'TOTAL_CLAIM_COST', 'RACE'))

def show_age_trends(filtered_df):
    st.title("📅 Age & Demographic Trends")
    # Line graph for the specific intersectional cohort
    age_trend = filtered_df.groupby(['AGE', 'GENDER'])['TOTAL_CLAIM_COST'].mean().reset_index()
    chart = alt.Chart(age_trend).mark_line(strokeWidth=3).encode(
        x='AGE:Q', 
        y='TOTAL_CLAIM_COST:Q',
        color='GENDER:N'
    ).properties(height=500)
    st.altair_chart(chart, use_container_width=True)
    st.info("This view highlights how aging costs diverge across gender identities within your selected demographic.")

def show_health_conditions(filtered_df):
    st.title("🩺 Intersectional Clinical Impact")
    top_problems = filtered_df.groupby('DESCRIPTION')['TOTAL_CLAIM_COST'].mean().sort_values(ascending=False).head(10).reset_index()
    
    chart = alt.Chart(top_problems).mark_bar().encode(
        x=alt.X('TOTAL_CLAIM_COST', title="Avg Cost ($)"),
        y=alt.Y('DESCRIPTION', sort='-x', title=None),
        color=alt.Color('DESCRIPTION', scale=alt.Scale(scheme='tableau20'), legend=None)
    ).properties(height=500)
    st.altair_chart(chart, use_container_width=True)
    st.info(get_interpretation(filtered_df, 'TOTAL_CLAIM_COST', 'DESCRIPTION'))

def show_ledger(filtered_df):
    st.title("📑 Intersectional Audit Ledger")
    search = st.text_input("Search filtered records...", "").lower()
    cols = ['PATIENT', 'GENDER', 'RACE', 'AGE', 'INCOME_TIER', 'INSURANCE_STATUS', 'DESCRIPTION', 'TOTAL_CLAIM_COST']
    existing = [c for c in cols if c in filtered_df.columns]
    
    if search:
        filtered_df = filtered_df[filtered_df.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)]
    st.dataframe(filtered_df[existing].head(500), use_container_width=True)

# --- 6. MAIN ROUTING ---
try:
    data = get_all_data()
    
    # 1. Apply Global Intersectional Filters
    filtered_data = sidebar_filters(data)
    
    # 2. Page Navigation
    page = st.sidebar.radio("Navigation", ["Overview", "Intersectional Analysis", "Age Trends", "Clinical Impact", "Audit Ledger"])

    if page == "Overview": show_overview()
    elif page == "Intersectional Analysis": show_multivariate_analysis(filtered_data)
    elif page == "Age Trends": show_age_trends(filtered_data)
    elif page == "Clinical Impact": show_health_conditions(filtered_data)
    elif page == "Audit Ledger": show_ledger(filtered_data)

except Exception as e:
    st.error("Deployment Error: Verify 'GENDER' and 'RACE' columns exist in data.")
    st.exception(e)
