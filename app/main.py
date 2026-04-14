import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

# --- 1. PAGE CONFIG & NGO STYLING ---
st.set_page_config(page_title="HEIP | NGO Health Equity OS", layout="wide", page_icon="🏥")

st.markdown("""
    <style>
    .main {background-color: #fdfdfd;}
    .big-header {
        background-color: #334155; 
        color: white; 
        padding: 30px; 
        border-radius: 10px; 
        font-size: 42px; 
        font-weight: 800; 
        text-align: center;
        margin-bottom: 25px;
    }
    .section-header {
        background-color: #ccfbf1; 
        color: #0f172a; 
        padding: 12px 20px; 
        border-radius: 8px; 
        font-size: 22px; 
        font-weight: 700; 
        margin-top: 25px; 
        margin-bottom: 15px; 
        border-left: 8px solid #2dd4bf;
    }
    .mission-box {
        background-color: #f8fafc; 
        padding: 25px; 
        border-radius: 15px; 
        border: 1px solid #e2e8f0; 
        margin-bottom: 25px;
    }
    </style>
    """, unsafe_allow_html=True)

NGO_PALETTE = px.colors.qualitative.Safe 

# --- 2. DATA ENGINE ---
@st.cache_data
def load_and_prep_data():
    root_path = Path(__file__).resolve().parent.parent
    data_dir = root_path / "data"
    
    p = pd.read_csv(data_dir / "patients.csv")
    for col in ['INCOME', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']:
        if col in p.columns:
            p[col] = pd.to_numeric(p[col].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce').fillna(0)
    
    p['INSURANCE_COVERAGE_PCT'] = (p['HEALTHCARE_COVERAGE'] / (p['HEALTHCARE_EXPENSES'] + 1) * 100).clip(0, 100)
    p['INCOME_TIER'] = p['INCOME'].apply(lambda x: 'Low' if x < 35000 else ('Middle' if x < 85000 else 'High'))
    
    e_files = list(data_dir.glob("encounters*.csv"))
    e = pd.concat([pd.read_csv(f) for f in e_files if f.stat().st_size > 0], ignore_index=True)
    e['START'] = pd.to_datetime(e['START'])
    e['YEAR'] = e['START'].dt.year 
    
    return pd.merge(e, p, left_on='PATIENT', right_on='Id', how='inner')

# --- 3. RANDOM FOREST GRID ENGINE (FIXED) ---
def get_rf_grid_projections(df, row_factor, col_factor, target_metric):
    rows = sorted(df[row_factor].unique())
    cols = sorted(df[col_factor].unique())
    all_projections = []

    for r in rows:
        for c in cols:
            subset = df[(df[row_factor] == r) & (df[col_factor] == c)]
            if len(subset['YEAR'].unique()) > 2:
                yearly = subset.groupby('YEAR')[target_metric].mean().reset_index()
                
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                X = yearly[['YEAR']].values
                y = yearly[target_metric].values
                rf.fit(X, y)
                
                future_years = np.array(range(yearly['YEAR'].max() + 1, 2031)).reshape(-1, 1)
                preds = rf.predict(future_years)
                
                # We must add row_factor and col_factor back into the dataframe here
                hist = yearly.assign(Status='Actual', Label=f"{r} | {c}")
                hist[row_factor] = r
                hist[col_factor] = c
                
                proj = pd.DataFrame({'YEAR': future_years.flatten(), target_metric: preds, 'Status': 'Projected', 'Label': f"{r} | {c}"})
                proj[row_factor] = r
                proj[col_factor] = c
                
                all_projections.append(pd.concat([hist, proj]))
    
    return pd.concat(all_projections) if all_projections else pd.DataFrame()

# --- 4. MAIN APP ---
try:
    raw_df = load_and_prep_data()

    st.sidebar.markdown("## 📊 Strategic Filters")
    sel_genders = st.sidebar.multiselect("Gender", options=sorted(raw_df['GENDER'].unique()), default=raw_df['GENDER'].unique())
    sel_races = st.sidebar.multiselect("Race", options=sorted(raw_df['RACE'].unique()), default=raw_df['RACE'].unique())
    sel_income = st.sidebar.multiselect("Income Tier", options=['Low', 'Middle', 'High'], default=['Low', 'Middle', 'High'])
    
    df = raw_df[(raw_df['GENDER'].isin(sel_genders)) & (raw_df['RACE'].isin(sel_races)) & (raw_df['INCOME_TIER'].isin(sel_income))]
    page = st.sidebar.radio("Navigation", ["Overview", "Interactive Map", "Population Comparison", "Predictive Grid (Random Forest)"])

    if page == "Overview":
        st.markdown('<div class="big-header">Health Equity Insights Platform</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">App Summary & Mission</div>', unsafe_allow_html=True)
        st.markdown('<div class="mission-box">Identifying <strong>Vertical Equity Gaps</strong> via intersectional data and Random Forest predictive modeling.</div>', unsafe_allow_html=True)
        with st.form("ngo_feedback"):
            st.write("Submit Feedback to NGO Analysts")
            f_name = st.text_input("Full Name")
            f_email = st.text_input("Organization Email")
            f_msg = st.text_area("Observations")
            if st.form_submit_button("Submit"):
                st.success("Thank you! Feedback Logged.")

    elif page == "Interactive Map":
        st.markdown('<div class="big-header">California Regional Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Expenditure Bubble Map</div>', unsafe_allow_html=True)
        
        map_stats = df.groupby('COUNTY').agg({'TOTAL_CLAIM_COST': 'mean', 'INCOME': 'mean', 'INSURANCE_COVERAGE_PCT': 'mean', 'LAT': 'mean', 'LON': 'mean'}).reset_index()
        fig_bub = px.scatter_mapbox(
            map_stats, lat="LAT", lon="LON", color="TOTAL_CLAIM_COST", 
            size="TOTAL_CLAIM_COST", hover_name="COUNTY",
            hover_data={'LAT': False, 'LON': False, 'TOTAL_CLAIM_COST': ':,.2f', 'INCOME': ':,.0f', 'INSURANCE_COVERAGE_PCT': ':.1f%'},
            color_continuous_scale="Teal", size_max=25, zoom=5, mapbox_style="carto-positron"
        )
        st.plotly_chart(fig_bub, use_container_width=True, key="bubble_map")

    elif page == "Population Comparison":
        st.markdown('<div class="big-header">Intersectional Comparison</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Demographic Equity Metrics</div>', unsafe_allow_html=True)
        metric = st.selectbox("Select Metric:", ['TOTAL_CLAIM_COST', 'INSURANCE_COVERAGE_PCT'])
        c1, c2 = st.columns(2)
        with c1:
            demo_a = st.selectbox("Group A by:", ['GENDER', 'RACE', 'INCOME_TIER'], key="a")
            st.plotly_chart(px.bar(df.groupby(demo_a)[metric].mean().reset_index(), x=demo_a, y=metric, color=demo_a, color_discrete_sequence=NGO_PALETTE), use_container_width=True, key="chart_a")
        with c2:
            demo_b = st.selectbox("Group B by:", ['INCOME_TIER', 'RACE', 'GENDER'], key="b")
            st.plotly_chart(px.bar(df.groupby(demo_b)[metric].mean().reset_index(), x=demo_b, y=metric, color=demo_b, color_discrete_sequence=NGO_PALETTE), use_container_width=True, key="chart_b")

    elif page == "Predictive Grid (Random Forest)":
        st.markdown('<div class="big-header">Intersectional Forecast Grid</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Random Forest Predictive Grid</div>', unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1: row_f = st.selectbox("Vertical Axis (Rows):", ['RACE', 'GENDER', 'INCOME_TIER'], index=0)
        with c2: col_f = st.selectbox("Horizontal Axis (Columns):", ['GENDER', 'RACE', 'INCOME_TIER'], index=1)
        with c3: target = st.selectbox("Metric to Forecast:", ['TOTAL_CLAIM_COST', 'INSURANCE_COVERAGE_PCT', 'INCOME'])

        grid_df = get_rf_grid_projections(raw_df, row_f, col_f, target)

        if not grid_df.empty:
            fig = px.line(
                grid_df, x="YEAR", y=target, color="Status",
                facet_row=row_f, facet_col=col_f,
                title=f"2030 Projections: {target.replace('_', ' ')}",
                color_discrete_map={'Actual': '#94a3b8', 'Projected': '#fb7185'},
                markers=True
            )
            fig.update_layout(height=250 * len(raw_df[row_f].unique()), margin=dict(t=80))
            st.plotly_chart(fig, use_container_width=True, key="grid_chart")
        else:
            st.warning("Insufficient data for Random Forest intersections.")

except Exception as e:
    st.error("🚨 System Update Required")
    st.exception(e)
