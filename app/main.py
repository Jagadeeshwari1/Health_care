import sys
import os
from pathlib import Path

# --- 1. DYNAMIC PATH INJECTION (THE FIX) ---
# This identifies '/mount/src/health_care' and adds it to the system path
# so that the statement 'from src.xxx' actually works.
current_dir = Path(__file__).resolve().parent
root_path = current_dir.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

# --- 2. NOW DO YOUR IMPORTS ---
import streamlit as st
import pandas as pd
import joblib

# These will no longer throw an ImportError
from src.data_processor import load_and_merge_data
from src.model import train_model

# --- 3. APP LOGIC ---
st.set_page_config(page_title="Health Equity Dashboard", layout="wide")

try:
    # Load and Clean Data
    df, report = load_and_merge_data()

    # Check for Model
    model_path = root_path / "models" / "cost_predictor.pkl"
    if not model_path.exists():
        with st.spinner("Training Predictive Engine..."):
            train_model(df)
    
    model = joblib.load(model_path)

    st.title("🏥 Health Equity Insights Dashboard")
    st.success("System Online: Data and Predictive Model Loaded.")
    
    # Display the report you generated in data_processor
    st.subheader("Vertical Equity Analysis (By Income and City)")
    st.dataframe(report)

except Exception as e:
    st.error("Deployment Configuration Error")
    st.exception(e)
