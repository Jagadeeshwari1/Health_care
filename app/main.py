import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

# --- 1. SETTINGS & STYLING ---
st.set_page_config(page_title="Intersectional Health Equity", layout="wide", page_icon="🏥")

st.markdown("""
    <style>
    .big-header {font-size: 50px !important; color: #1E3A8A; font-weight: 800; text-align: center; margin-bottom: 30px; border-bottom: 5px solid #FF4B4B;}
    .sub-header {background-color: #1E3A8A; color: white; padding: 12px; border-radius: 8px; font-size: 20px; margin-top: 20px;}
    .card {background-color: #f0f4f8; border: 1px solid #d1d5db; padding: 20px; border-radius: 10px; height: 100%;}
    </style>
    """, unsafe_allow_html=True)

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
        if c in p.columns:
            p[c] = pd.to_numeric(p[c].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce').fillna(0)
    
    p['INCOME_TIER'] = p['INCOME'].apply(lambda x: 'Low' if x < 35000 else ('Middle' if x < 85000 else 'High'))
    p['INSURANCE_STATUS'] = p['HEALTHCARE_COVERAGE'].apply(
        lambda x: 'Under-Insured' if x < 5000 else ('Standard' if x < 15000 else 'Premium')
    )
    
    # --- SAFE ENCOUNTER LOADING ---
    e_path = data_dir / "encounters_part_1.csv"
    if not e_path.exists() or e_path.stat().st_size == 0:
        e_path = data_dir / "encounters.csv" # Fallback
        
    e = pd.read_csv(e_path)
    df = pd.merge(e, p, left_on='PATIENT', right_on='Id', how='inner')
    return df

def get_interpretation(data, metric, group_col):
    if data.empty: return "No data available for this specific intersection."
    stats = data.groupby(group_col)[metric].mean().sort_values(ascending=False)
    return f"💡 **Intersectional Insight:** The **{stats.index[0]}** cohort in this filtered group shows the highest average burden at **${stats.iloc[0]:,.2f}**."

# --- 3. MAIN APP ---
try:
    df = get_all_data()

    # --- SIDEBAR: THE INTERSECTIONAL ENGINE ---
    st.sidebar.title("🎛️ Intersectional Filters")
    
    age_range = st.sidebar.slider("Age Bracket", 0, 100, (0, 100))
    
    # Ensure columns exist before filtering
    genders = sorted(df['GENDER'].unique()) if 'GENDER' in df.columns else ['Unknown']
    sel_gender = st.sidebar.multiselect("Gender", genders, default=genders)
    
    races = sorted(df['RACE'].unique()) if 'RACE' in df.columns else ['Unknown']
    sel_race = st.sidebar.multiselect("Race/Ethnicity", races, default=races)
    
    tiers = ['Low', 'Middle', 'High']
    sel_income = st.sidebar.multiselect("Income Tier", tiers, default=tiers)

    # APPLY FILTERS
    mask = (
        (df['AGE'] >= age_range[0]) & (df['AGE'] <= age_range[1]) &
        (df['GENDER'].isin(sel_gender)) &
        (df['RACE'].isin(sel_race)) &
        (df['INCOME_TIER'].isin(sel_income))
    )
    f_df = df[mask]

    # NAVIGATION
    page = st.sidebar.radio("Navigation", ["Overview", "Intersectional Analysis", "Age Trends", "Clinical Impact", "Audit Ledger"])

    if page == "Overview":
        st.markdown('<p class="big-header">HEALTH EQUITY INSIGHTS PLATFORM</p>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background-color: #eef2ff; padding: 25px; border-radius: 15px; border-left: 8px solid #1E3A8A;">
        <h3>Project Mission</h3>
        This application identifies <strong>Vertical Equity Gaps</strong> across the healthcare landscape. 
        By intersecting data across <b>Gender, Race, Age, and Wealth</b>, we allow for targeted resource allocation.
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="card"><p class="sub-header">🎯 Decision Makers</p>'
                        '<ul><li>Public Health Officials</li><li>Policy Legislators</li></ul></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card"><p class="sub-header">👥 End Users</p>'
                        '<ul><li>Clinical Data Analysts</li><li>Social Workers</li></ul></div>', unsafe_allow_html=True)

    elif page == "Intersectional Analysis":
        st.title("🧩 Multi-Factor Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Cost by Gender & Income")
            c = alt.Chart(f_df).mark_bar().encode(
                x='GENDER:N', y='mean(TOTAL_CLAIM_COST):Q', 
                color=alt.Color('INCOME_TIER', scale=alt.Scale(domain=['High', 'Middle', 'Low'], range=['#FF4B4B', '#FFD700', '#2E8B57'])),
                column='INCOME_TIER:N'
            ).properties(width=120, height=350)
            st.altair_chart(c)
            st.info(get_interpretation(f_df, 'TOTAL_CLAIM_COST', 'INCOME_TIER'))
        with col2:
            st.subheader("Cost by Race")
            rc = alt.Chart(f_df).mark_bar(color='#4B0082').encode(
                x='mean(TOTAL_CLAIM_COST):Q', y=alt.Y('RACE:N', sort='-x')
            ).properties(height=350)
            st.altair_chart(rc, use_container_width=True)
            st.info(get_interpretation(f_df, 'TOTAL_CLAIM_COST', 'RACE'))

    elif page == "Age Trends":
        st.title("📅 Age Trends")
        trend = f_df.groupby(['AGE', 'GENDER'])['TOTAL_CLAIM_COST'].mean().reset_index()
        line = alt.Chart(trend).mark_line(strokeWidth=3).encode(x='AGE:Q', y='TOTAL_CLAIM_COST:Q', color='GENDER:N')
        st.altair_chart(line, use_container_width=True)
        st.info(get_interpretation(f_df, 'TOTAL_CLAIM_COST', 'AGE'))

    elif page == "Clinical Impact":
        st.title("🩺 Clinical Impact")
        probs = f_df.groupby('DESCRIPTION')['TOTAL_CLAIM_COST'].mean().sort_values(ascending=False).head(10).reset_index()
        st.bar_chart(probs, x='DESCRIPTION', y='TOTAL_CLAIM_COST', color="#1F77B4")
        st.info(get_interpretation(f_df, 'TOTAL_CLAIM_COST', 'DESCRIPTION'))

    elif page == "Audit Ledger":
        st.title("📑 Ledger")
        st.dataframe(f_df[['PATIENT', 'GENDER', 'RACE', 'AGE', 'INCOME_TIER', 'DESCRIPTION', 'TOTAL_CLAIM_COST']].head(100))

except Exception as e:
    st.error("🚨 Critical Error: Check your data files.")
    st.exception(e)
