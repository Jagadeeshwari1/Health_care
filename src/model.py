import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
from pathlib import Path

def train_model(df):
    features = ['AGE', 'INCOME', 'HEALTHCARE_COVERAGE']
    target = 'TOTAL_CLAIM_COST'

    # Clean and Force Numeric
    for col in features + [target]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(r'[\$,]', '', regex=True), 
            errors='coerce'
        ).fillna(0)

    X = df[features]
    y = df[target]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save to /models folder
    root_path = Path(__file__).parents[1]
    model_dir = root_path / "models"
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, model_dir / "cost_predictor.pkl")
    return True
