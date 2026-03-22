import sys
from pathlib import Path

# Fix path
current_dir = Path(__file__).resolve().parent
root = current_dir.parent
sys.path.insert(0, str(root))

import streamlit as st
from src.data_processor import load_data
from src.visuals import show_cost_analysis, show_city_analysis
from src.model import train_model, load_model

st.set_page_config(page_title="Health Equity Dashboard", layout="wide")

st.title("🏥 Health Equity & Cost Burden Dashboard")

# Load data
df, report = load_data()

# Train model once
if not (root / "models/model.pkl").exists():
    train_model(df)

model = load_model()

# Sidebar
city = st.sidebar.selectbox("Select City", sorted(df['CITY'].dropna().unique()))

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    show_cost_analysis(report)
    show_city_analysis(df, city)

with col2:
    st.subheader("🔮 Predict Cost")

    age = st.number_input("Age", 0, 100, 40)
    income = st.number_input("Income", 0, 200000, 30000)
    coverage = st.number_input("Coverage", 0, 200000, 10000)

    if st.button("Predict"):
        if model:
            pred = model.predict([[age, income, coverage]])[0]
            st.success(f"Estimated Cost: ${pred:,.2f}")
           
