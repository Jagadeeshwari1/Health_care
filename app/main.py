import streamlit as st
import pandas as pd
import altair as alt
import glob
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

# --- 2. THE "SMART" DATA ENGINE ---
@st.cache_data
def get_all_data():
    # Use absolute path resolution to avoid FileNotFoundError
    root_path = Path(__file__).resolve().parent.parent
    data_dir = root_path / "data"
    
    # 1. Load Patients
    p_path = data_dir / "patients.csv"
    if not p_path.exists():
        raise FileNotFoundError(f"Could not find patients.csv at {p_path}")
    p = pd.read_csv(p_path)
    
    # Clean Patients
    p['BIRTHDATE'] = pd.to_datetime(p['BIRTHDATE'])
    p['AGE'] = (pd.Timestamp.today() - p['BIRTHDATE']).dt.days // 365
    for c in ['INCOME', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']:
        if c in p.columns:
            p[c] = pd.to_numeric(p[c].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce').fillna(0)
    p['INCOME_TIER'] = p['INCOME'].apply(lambda x: 'Low' if x < 35000 else ('Middle' if x < 85000 else 'High'))

    # 2. SMART ENCOUNTER LOADING (Finds any file starting with 'encounters')
    encounter_files = list(data_dir.glob("encounters*.csv"))
    if not encounter_files:
        raise FileNotFoundError(f"No encounter files found in {data_dir}")
    
    # Merge all encounter parts if they exist
    e_list = [pd.read_csv(f) for f in encounter_files]
    e = pd.concat(e_list, ignore_index=True)
    
    # 3. Final Merge
    df = pd.merge(e, p, left_on='PATIENT', right_on='Id', how='inner')
    return df

def get_interpretation(data, metric, group_col):
    if data.empty: return "No data available for this specific intersection."
    stats = data.groupby(group_col)[metric].mean().sort_values(ascending=False)
    top_group = stats.index[0]
    top_val = stats.iloc[0]
    return f"💡 **Intersectional Insight:** The **{top_group}** cohort in this filtered group shows the highest average burden at **${top_val:,.2f}**."

# --- 3. MAIN APP ---
try:
    df = get_all_data()

    # --- GLOBAL SIDEBAR (INTERSECTIONAL ENGINE) ---
    st.sidebar.title("🎛️ Multi-Factor Filters")
    
    age_range = st.sidebar.slider("Age Bracket", 0, 100, (0, 100))
    
    # Column Check for Gender and Race
    g_cols = [c for c in ['GENDER', 'Gender'] if c in df.columns]
    g_col = g_cols[0] if g_cols else None
    
    r_cols = [c for c in ['RACE', 'Race'] if c in df.columns]
    r_col = r_cols[0] if r_cols else None

    sel_gender = st.sidebar.multiselect("Gender Focus", sorted(df[g_col].unique())) if g_col else []
    sel_race = st.sidebar.multiselect("Race Focus", sorted(df[r_col].unique())) if r_col else []
    
    tiers = ['Low', 'Middle', 'High']
    sel_income = st.sidebar.multiselect("Income Tier Focus", tiers, default=tiers)
    
    counties = sorted(df['COUNTY'].unique()) if 'COUNTY' in df.columns else []
    sel_county = st.sidebar.multiselect("County/Region Focus", counties, default=counties[:10])

    # APPLY FILTERS
    mask = (df['AGE'] >= age_range[0]) & (df['AGE'] <= age_range[1])
    if g_col and sel_gender: mask &= df[g_col].isin(sel_gender)
    if r_col and sel_race: mask &= df[r_col].isin(sel_race)
    if sel_income: mask &= df['INCOME_TIER'].isin(sel_income)
    if sel_county: mask &= df['COUNTY'].isin(sel_county)
    
    f_df = df[mask]

    # NAVIGATION
    page = st.sidebar.radio("Navigation", ["Overview", "Intersectional Analysis", "Age Trends", "Geography Analysis", "Clinical Impact", "Data Ledger"])

    if page == "Overview":
        st.markdown('<p class="big-header">HEALTH EQUITY INSIGHTS PLATFORM</p>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background-color: #eef2ff; padding: 25px; border-radius: 15px; border-left: 8px solid #1E3A8A; margin-bottom: 30px;">
        <h3>Project Mission</h3>
        This application identifies <strong>Vertical Equity Gaps</strong> across the healthcare landscape. 
        By intersecting clinical data with multi-dimensional socio-economic indicators—including age, race, gender, and wealth—we empower stakeholders to make informed, intersectional resource allocations.
        <br><br><strong><a href="https://github.com/thanujakalla/final_project_version-2">🔗 Detailed Project Documentation</a></strong>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="card"><p class="sub-header">🎯 Decision Makers</p>'
                        '<ul><li><b>Public Health Officials:</b> Global resource distribution</li>'
                        '<li><b>Policy Analysts:</b> Equity legislation</li></ul></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card"><p class="sub-header">👥 End Users</p>'
                        '<ul><li><b>Data Analysts:</b> Multi-factor clinical trends</li>'
                        '<li><b>Advocacy Groups:</b> Targeted community support</li></ul></div>', unsafe_allow_html=True)

    elif page == "Intersectional Analysis":
        st.title("🧩 Multi-Factor Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Cost by Gender & Income")
            c = alt.Chart(f_df).mark_bar().encode(
                x=alt.X(f'{g_col}:N', title=None), y='mean(TOTAL_CLAIM_COST):Q', 
                color=alt.Color('INCOME_TIER', scale=alt.Scale(domain=['High', 'Middle', 'Low'], range=['#FF4B4B', '#FFD700', '#2E8B57'])),
                column='INCOME_TIER:N'
            ).properties(width=120, height=350)
            st.altair_chart(c)
            st.info(get_interpretation(f_df, 'TOTAL_CLAIM_COST', 'INCOME_TIER'))
        with col2:
            st.subheader("Cost by Race")
            rc = alt.Chart(f_df).mark_bar(color='#4B0082').encode(
                x='mean(TOTAL_CLAIM_COST):Q', y=alt.Y(f'{r_col}:N', sort='-x', title=None)
            ).properties(height=350)
            st.altair_chart(rc, use_container_width=True)
            st.info(get_interpretation(f_df, 'TOTAL_CLAIM_COST', r_col))

    elif page == "Age Trends":
        st.title("📅 Age Trends")
        trend = f_df.groupby(['AGE', g_col])['TOTAL_CLAIM_COST'].mean().reset_index()
        line = alt.Chart(trend).mark_line(strokeWidth=3).encode(x='AGE:Q', y='TOTAL_CLAIM_COST:Q', color=f'{g_col}:N')
        st.altair_chart(line, use_container_width=True)
        st.info(get_interpretation(f_df, 'TOTAL_CLAIM_COST', 'AGE'))

    elif page == "Geography Analysis":
        st.title("🌍 Regional Cost Comparison")
        county_stats = f_df.groupby('COUNTY')['TOTAL_CLAIM_COST'].mean().sort_values(ascending=False).head(10).reset_index()
        g_chart = alt.Chart(county_stats).mark_bar(color='#10B981').encode(
            x=alt.X('TOTAL_CLAIM_COST:Q', title="Avg Cost ($)"), y=alt.Y('COUNTY:N', sort='-x')
        ).properties(height=450)
        st.altair_chart(g_chart, use_container_width=True)
        st.info(get_interpretation(county_stats, 'TOTAL_CLAIM_COST', 'COUNTY'))

    elif page == "Clinical Impact":
        st.title("🩺 Clinical Impact")
        probs = f_df.groupby('DESCRIPTION')['TOTAL_CLAIM_COST'].mean().sort_values(ascending=False).head(10).reset_index()
        st.bar_chart(probs, x='DESCRIPTION', y='TOTAL_CLAIM_COST', color="#1F77B4")
        st.info(get_interpretation(f_df, 'TOTAL_CLAIM_COST', 'DESCRIPTION'))

    elif page == "Data Ledger":
        st.title("📑 Ledger")
        st.dataframe(f_df.head(100))

except Exception as e:
    st.error("🚨 System Error")
    st.write(e)
