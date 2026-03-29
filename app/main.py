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
        if c in p.columns:
            p[c] = pd.to_numeric(p[c].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce').fillna(0)
    
    # Professor's Request: Insurance Coverage %
    p['INSURANCE_COVERAGE_PCT'] = (p['HEALTHCARE_COVERAGE'] / (p['HEALTHCARE_EXPENSES'] + 1) * 100).clip(0, 100)
    
    # Load and Merge Encounters (Handles multiple parts)
    encounter_files = list(data_dir.glob("encounters*.csv"))
    if not encounter_files:
        st.error("No encounter data found in /data folder.")
        return pd.DataFrame()
        
    e = pd.concat([pd.read_csv(f) for f in encounter_files], ignore_index=True)
    e['START'] = pd.to_datetime(e['START'])
    e['YEAR'] = e['START'].dt.year
    
    df = pd.merge(e, p, left_on='PATIENT', right_on='Id', how='inner')
    return df

# --- 3. PAGE FUNCTIONS ---

def show_intro():
    st.title("🏥 Health Equity Insights Platform")
    st.markdown("""
    ### Welcome
    This platform identifies **Vertical Equity Gaps**—disparities in healthcare access and cost based on socio-economic status. 
    We use intersectional data to help decision-makers allocate resources where they are needed most.
    """)
    st.info("💡 **How to use:** Navigate using the sidebar to explore the California Map, Compare Demographic groups, or use the Predictive Tool to see future trends.")

def show_map_page(df):
    st.header("📍 Interactive California Health Map")
    st.markdown("> **Non-Technical Summary:** This map colors California counties based on healthcare costs. Darker colors represent higher financial burdens. Hover over any county to see average income and insurance coverage.")

    # Aggregate by County
    map_data = df.groupby('COUNTY').agg({
        'TOTAL_CLAIM_COST': 'mean',
        'INCOME': 'mean',
        'INSURANCE_COVERAGE_PCT': 'mean'
    }).reset_index()

    # Create the Map (Choropleth)
    fig = px.choropleth(
        map_data,
        locations='COUNTY',
        locationmode='USA-states', # Note: For exact county borders, a GeoJSON is usually needed, but this provides the hover interactivity.
        color='TOTAL_CLAIM_COST',
        hover_name='COUNTY',
        hover_data={'INCOME': ':,.2f', 'INSURANCE_COVERAGE_PCT': ':.1f%'},
        color_continuous_scale="Viridis",
        scope="usa",
        title="Average Claims Cost by County"
    )
    # Focus map on California region roughly
    fig.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # Interactive Click-down (Professor's Request #1.2)
    st.divider()
    selected_county = st.selectbox("Detailed County View: Select a County to generate specific graphs", options=sorted(df['COUNTY'].unique()))
    county_df = df[df['COUNTY'] == selected_county]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Claims Cost", f"${county_df['TOTAL_CLAIM_COST'].mean():,.2f}")
    c2.metric("Avg Income", f"${county_df['INCOME'].mean():,.2f}")
    c3.metric("Insurance Coverage %", f"{county_df['INSURANCE_COVERAGE_PCT'].mean():.1f}%")

def show_compare_page(df):
    st.header("⚖️ Population Comparison Tool")
    st.markdown("> **Non-Technical Summary:** Use this page to compare two or more groups side-by-side. For example, you can see how much more 'Low Income' groups pay compared to 'High Income' groups.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        factor = st.selectbox("Compare By:", ['INCOME_TIER', 'GENDER', 'RACE'])
        metric = st.selectbox("Metric:", ['TOTAL_CLAIM_COST', 'INCOME', 'INSURANCE_COVERAGE_PCT'])
    
    with col2:
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(factor, sort='-y'),
            y=alt.Y(f'mean({metric}):Q', title=f"Average {metric}"),
            color=alt.Color(factor, scale=alt.Scale(scheme='tableau10')),
            tooltip=[factor, f'mean({metric})']
        ).properties(height=450)
        st.altair_chart(chart, use_container_width=True)

def show_predictive_page(df):
    st.header("🔮 Future Trend Projections")
    st.markdown("> **Non-Technical Summary:** This tool uses mathematical models to look at our data from the past to the present, then draws a line into the future to predict where costs are headed by 2030.")
    
    dep_var = st.selectbox("Select Projection Variable:", ['TOTAL_CLAIM_COST', 'INCOME', 'INSURANCE_COVERAGE_PCT'])
    
    # Prepare Data
    yearly = df.groupby('YEAR')[dep_var].mean().reset_index()
    X = yearly[['YEAR']].values
    y = yearly[dep_var].values
    
    # Train Model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict to 2030
    future_years = np.array([[2026], [2027], [2028], [2029], [2030]])
    future_preds = model.predict(future_years)
    
    # Combine for Charting
    hist_df = pd.DataFrame({'Year': yearly['YEAR'], 'Value': y, 'Status': 'Past/Present'})
    proj_df = pd.DataFrame({'Year': future_years.flatten(), 'Value': future_preds, 'Status': 'Future Projection'})
    combined = pd.concat([hist_df, proj_df])
    
    chart = alt.Chart(combined).mark_line(point=True).encode(
        x=alt.X('Year:O'),
        y=alt.Y('Value:Q', title=f"Estimated {dep_var}"),
        color='Status',
        strokeDash='Status'
    ).properties(height=500)
    
    st.altair_chart(chart, use_container_width=True)
    st.success(f"By 2030, the model projects {dep_var.replace('_',' ')} will reach approximately **${future_preds[-1]:,.2f}**.")

# --- 4. MAIN ROUTING ---
try:
    data = get_all_data()
    if not data.empty:
        page = st.sidebar.radio("Navigation", ["Intro", "Interactive Map", "Compare Groups", "Predictive Tool"])

        if page == "Intro": show_intro()
        elif page == "Interactive Map": show_map_page(data)
        elif page == "Compare Groups": show_compare_page(data)
        elif page == "Predictive Tool": show_predictive_page(data)
except Exception as e:
    st.error("System Error: Please ensure all data columns (GENDER, RACE, COUNTY) exist.")
    st.exception(e)
    
