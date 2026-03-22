from sklearn.ensemble import RandomForestRegressor
import joblib
from pathlib import Path
import pandas as pd

def train_model(df):
    # ✅ Select only required columns
    df = df[['AGE', 'INCOME', 'HEALTHCARE_COVERAGE', 'HEALTHCARE_EXPENSES']]

    # ✅ Convert to numeric (VERY IMPORTANT)
    df = df.apply(pd.to_numeric, errors='coerce')

    # ✅ Drop missing values
    df = df.dropna()

    # 🚨 Safety check
    if df.empty:
        raise ValueError("No valid data available after cleaning")

    X = df[['AGE', 'INCOME', 'HEALTHCARE_COVERAGE']]
    y = df['HEALTHCARE_EXPENSES']

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    root = Path(__file__).parents[1]
    model_path = root / "models" / "model.pkl"
    model_path.parent.mkdir(exist_ok=True)

    joblib.dump(model, model_path)
