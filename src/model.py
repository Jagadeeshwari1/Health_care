from sklearn.ensemble import RandomForestRegressor
import joblib
from pathlib import Path

def train_model(df):
    df = df.dropna()

    X = df[['AGE', 'INCOME', 'HEALTHCARE_COVERAGE']]
    y = df['HEALTHCARE_EXPENSES']

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    root = Path(__file__).parents[1]
    model_path = root / "models" / "model.pkl"
    model_path.parent.mkdir(exist_ok=True)

    joblib.dump(model, model_path)

def load_model():
    root = Path(__file__).parents[1]
    model_path = root / "models" / "model.pkl"

    if model_path.exists():
        return joblib.load(model_path)
    return None
