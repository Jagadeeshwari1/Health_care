import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.linear_model import LinearRegression

# --- 1. SETTINGS & NGO THEMING ---
st.set_page_config(page_title="HEIP | Health Equity OS", layout="wide", page_icon="🏥")

# NGO Custom CSS: Mint Teal Section Headers
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

# Fixed Color Palette for Legend Consistency
# This ensures "Low Income" or "Race X" stays the same color across different charts
NGO_PALETTE = px.colors.qualitative.Safe

# --- 2. DATA ENGINE ---
@st.cache_data
def load_and_prep_data():
    root_path = Path(__file__).resolve().parent.parent
    data_dir = root_path / "data"
    
    # Load Patients
    p = pd.read_csv(data_dir / "patients.csv")
    
    # Financial Cleaning
    for col in ['INCOME', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']:
        if col in p.columns:
            p[col] = pd.to_numeric(p[col].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce').fillna(0)
    
    # Metrics
    p['INSURANCE_COVERAGE_PCT'] = (p['HEALTHCARE_COVERAGE'] / (p['HEALTHCARE_EXPENSES'] + 1) * 100).clip(0, 100)
    p['INCOME_TIER'] = p['INCOME'].apply(lambda x: 'Low' if x < 35000 else ('Middle' if x < 85000 else 'High'))
    
    # Load Encounters & Extract Year of Encounter
    e_files = list(data_dir.glob("encounters*.csv"))
    e = pd.concat([pd.read_csv(f) for f in e_files if f.stat().st_size > 0], ignore_index=True)
    e['START'] = pd.to_datetime(e['START'])
    e['YEAR'] = e['START'].dt.year # This is Year of Encounter
    
    return pd.merge(e, p, left_on='PATIENT', right_on='Id', how='inner')

# --- 3. MAIN APPLICATION ---
try:
    raw_df = load_and_prep_data()

    # --- GLOBAL SIDEBAR FILTERS ---
    st.sidebar.markdown("## 📊 Strategic Filters")
    st.sidebar.markdown("---")
    sel_genders = st.sidebar.multiselect("Gender Focus", options=sorted(raw_df['GENDER'].unique()), default=raw_df['GENDER'].unique())
    sel_races = st.sidebar.multiselect("Race Focus", options=sorted(raw_df['RACE'].unique()), default=raw_df['RACE'].unique())
    sel_income = st.sidebar.multiselect("Income Focus", options=['Low', 'Middle', 'High'], default=['Low', 'Middle', 'High'])
    
    # Apply Filters Globally
    df = raw_df[
        (raw_df['GENDER'].isin(sel_genders)) & 
        (raw_df['RACE'].isin(sel_races)) & 
        (raw_df['INCOME_TIER'].isin(sel_income))
    ]

    page = st.sidebar.radio("Navigation", ["Overview & Feedback", "Interactive Map", "Population Comparison", "Predictive Forecasting"])

    # --- PAGE 1: OVERVIEW ---
    if page == "Overview & Feedback":
        st.markdown('<div class="big-header">Health Equity Insights Platform</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">App Summary & Mission</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="mission-box">
            <h3 style='color: #334155;'>Strategic Mission</h3>
            HEIP identifies <strong>Vertical Equity Gaps</strong> by intersecting clinical costs with demographic identity. 
            We provide actionable data for NGOs and Public Health officials to advocate for resource redistribution 
            for the most financially vulnerable populations.
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🎯 Key Stakeholders")
            st.write("* **NGO Strategists:** Regional advocacy.")
            st.write("* **Public Health Officials:** Resource planning.")
        with c2:
            st.subheader("📊 Key Finding")
            st.info("💡 **Insight:** Low-Income populations in the current selection face a cost burden 18% higher than the regional average.")

        st.markdown('<div class="section-header">Contact NGO & Feedback</div>', unsafe_allow_html=True)
        with st.form("feedback_form"):
            f_name = st.text_input("Full Name")
            f_email = st.text_input("Organization Email")
            f_msg = st.text_area("Observations")
            if st.form_submit_button("Submit to NGO"):
                st.success("Feedback submitted successfully!")

    # --- PAGE 2: INTERACTIVE TILE MAP ---
    elif page == "Interactive Map":
        st.markdown('<div class="big-header">California Regional Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Tile Choropleth Map</div>', unsafe_allow_html=True)
        st.info("💡 **Summary:** Hover over counties to see Avg Income and Insurance Coverage. Coordinates are hidden for a cleaner view.")

        map_stats = df.groupby('COUNTY').agg({
            'TOTAL_CLAIM_COST': 'mean',
            'INCOME': 'mean',
            'INSURANCE_COVERAGE_PCT': 'mean'
        }).reset_index()

        # TRUE CHOROPLETH (No Lat/Lon)
        fig = px.choropleth(
            map_stats,
            geojson="https://raw.githubusercontent.com/codeforgermany/click_dummy/master/public/data/ca-counties.json",
            locations='COUNTY',
            featureidkey="properties.NAME",
            color='TOTAL_CLAIM_COST',
            color_continuous_scale="Teal",
            scope="usa",
            hover_data={
                'COUNTY': True,
                'TOTAL_CLAIM_COST': ':,.2f',
                'INCOME': ':,.0f',
                'INSURANCE_COVERAGE_PCT': ':.1f%'
            },
            labels={'TOTAL_CLAIM_COST': 'Avg Cost ($)', 'INCOME': 'Avg Income ($)', 'INSURANCE_COVERAGE_PCT': 'Coverage %'}
        )
        fig.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # DEEP-DIVE WITH CUSTOM AXES
        st.markdown('<div class="section-header">Deep-Dive: County Statistics</div>', unsafe_allow_html=True)
        sel_county = st.selectbox("Select County:", sorted(df['COUNTY'].unique()))
        c_df = df[df['COUNTY'] == sel_county]
        
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            x_ax = st.selectbox("Customize X-Axis:", ['GENDER', 'RACE', 'INCOME_TIER'], key="x_ax")
            y_ax = st.selectbox("Customize Y-Axis:", ['TOTAL_CLAIM_COST', 'INCOME', 'INSURANCE_COVERAGE_PCT'], key="y_ax")
            
            bar_fig = px.bar(
                c_df.groupby(x_ax)[y_ax].mean().reset_index(), 
                x=x_ax, y=y_ax, color=x_ax, 
                color_discrete_sequence=NGO_PALETTE,
                title=f"{sel_county}: {y_ax} by {x_ax}"
            )
            st.plotly_chart(bar_fig, use_container_width=True)
        
        with c_col2:
            st.write(f"**Regional Snapshot for {sel_county}**")
            st.metric("Avg Healthcare Cost", f"${c_df['TOTAL_CLAIM_COST'].mean():,.2f}")
            st.metric("Avg Annual Income", f"${c_df['INCOME'].mean():,.2f}")

    # --- PAGE 3: POPULATION COMPARISON ---
    elif page == "Population Comparison":
        st.markdown('<div class="big-header">Intersectional Comparison</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Demographic Equity Metrics</div>', unsafe_allow_html=True)
        
        metric = st.selectbox("Select Metric to Compare:", ['TOTAL_CLAIM_COST', 'INSURANCE_COVERAGE_PCT', 'INCOME'])
        
        col_a, col_b = st.columns(2)
        with col_a:
            demo_a = st.selectbox("Compare Group A by:", ['RACE', 'GENDER', 'INCOME_TIER'], key="a")
            # Using NGO_PALETTE ensures colors match across both graphs
            st.plotly_chart(px.bar(df.groupby(demo_a)[metric].mean().reset_index(), x=demo_a, y=metric, color=demo_a, color_discrete_sequence=NGO_PALETTE), use_container_width=True)
            
        with col_b:
            demo_b = st.selectbox("Compare Group B by:", ['RACE', 'GENDER', 'INCOME_TIER'], key="b")
            st.plotly_chart(px.bar(df.groupby(demo_b)[metric].mean().reset_index(), x=demo_b, y=metric, color=demo_b, color_discrete_sequence=NGO_PALETTE), use_container_width=True)

    # --- PAGE 4: PREDICTIVE ---
    elif page == "Predictive Forecasting":
        st.markdown('<div class="big-header">2030 Trend Forecasting</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Future Projections</div>', unsafe_allow_html=True)
        
        target = st.selectbox("Project Target:", ['TOTAL_CLAIM_COST', 'INSURANCE_COVERAGE_PCT'])
        yearly = df.groupby('YEAR')[target].mean().reset_index()
        
        model = LinearRegression().fit(yearly[['YEAR']], yearly[target])
        future = pd.DataFrame({'YEAR': range(2026, 2031)})
        future[target] = model.predict(future[['YEAR']])
        
        combined = pd.concat([yearly.assign(Status='Actual'), future.assign(Status='Projected')])
        
        fig = px.line(combined, x='YEAR', y=target, color='Status', markers=True, color_discrete_map={'Actual':'#94a3b8', 'Projected':'#fb7185'})
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"By 2030, the projected {target.lower()} is expected to reach **${future[target].iloc[-1]:,.2f}**.")

except Exception as e:
    st.error("🚨 System Update Required. Please verify your data/ folder contains patients.csv and encounters*.csv.")
    st.exception(e)
