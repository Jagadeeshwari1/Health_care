import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

# --- 1. SETTINGS & STYLING ---
st.set_page_config(page_title="Intersectional Health Equity", layout="wide", page_icon="🏥")

# Custom CSS for the Large Header you requested
st.markdown("""
    <style>
    .big-header {font-size: 50px !important; color: #1E3A8A; font-weight: 800; text-align: center; margin-bottom: 30px; border-bottom: 5px solid #FF4B4B;}
    .sub-header {background-color: #1E3A8A; color: white; padding: 12px; border-radius: 8px; font-size: 20px; margin-top: 20px;}
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
    
    # Cleaning and Tiers
    for c in ['INCOME', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']:
        p[c] = pd.to_numeric(p[c].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce').fillna(0)
    
    p['INCOME_TIER'] = p['INCOME'].apply(lambda x: 'Low' if x < 35000 else ('Middle' if x < 85000 else 'High'))
    
    # Load and Merge
    e = pd.read_csv(data_dir / "encounters_part_1.csv")
    df = pd.merge(e, p, left_on='PATIENT', right_on='Id', how='inner')
    return df

# --- 3. PAGE NAVIGATION & GLOBAL FILTERS ---
try:
    df = get_all_data()

    # GLOBAL SIDEBAR FOR INTERSECTIONALITY
    st.sidebar.title("🎛️ Intersectional Filters")
    st.sidebar.info("Select multiple factors to analyze specific cohorts.")

    # Multi-factor filters
    age_range = st.sidebar.slider("Age Bracket", 0, 100, (0, 100))
    
    # Re-introducing Gender & Race
    genders = sorted(df['GENDER'].unique()) if 'GENDER' in df.columns else []
    sel_gender = st.sidebar.multiselect("Gender Identity", genders, default=genders)
    
    races = sorted(df['RACE'].unique()) if 'RACE' in df.columns else []
    sel_race = st.sidebar.multiselect("Race/Ethnicity", races, default=races)
    
    income_tiers = ['Low', 'Middle', 'High']
    sel_income = st.sidebar.multiselect("Income Tier", income_tiers, default=income_tiers)

    # Applying the Intersectional Mask
    filtered_df = df[
        (df['AGE'] >= age_range[0]) & (df['AGE'] <= age_range[1]) &
        (df['GENDER'].isin(sel_gender)) &
        (df['RACE'].isin(sel_race)) &
        (df['INCOME_TIER'].isin(sel_income))
    ]

    # TAB NAVIGATION
    page = st.sidebar.radio("Navigation", ["Overview", "Intersectional Analysis", "Age Trends", "Clinical Impact", "Audit Ledger"])

    if page == "Overview":
        st.markdown('<p class="big-header">HEALTH EQUITY INSIGHTS PLATFORM</p>', unsafe_allow_html=True)
        st.markdown("### Project Mission: Identifying Vertical Equity Gaps across the healthcare landscape.")
        st.write("Analyze the intersection of Gender, Race, and Wealth to identify underserved populations.")

    elif page == "Intersectional Analysis":
        st.title("🧩 Multi-Factor Analysis")
        # Multi-factor Graph: Gender + Income
        chart = alt.Chart(filtered_df).mark_bar().encode(
            x='GENDER:N',
            y='mean(TOTAL_CLAIM_COST):Q',
            color=alt.Color('INCOME_TIER', scale=alt.Scale(domain=['High', 'Middle', 'Low'], range=['#FF4B4B', '#FFD700', '#2E8B57'])),
            column='INCOME_TIER:N'
        ).properties(width=150, height=400)
        st.altair_chart(chart)
        
        # Race Analysis
        race_chart = alt.Chart(filtered_df).mark_bar(color='#4B0082').encode(
            x='mean(TOTAL_CLAIM_COST):Q',
            y=alt.Y('RACE:N', sort='-x')
        )
        st.altair_chart(race_chart, use_container_width=True)

    # ... (Other pages follow the same pattern using filtered_df)

except Exception as e:
    st.error("Error Loading Dashboard. Check if GENDER and RACE columns exist in patients.csv.")
    st.exception(e)
