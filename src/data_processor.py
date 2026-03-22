import pandas as pd
from pathlib import Path

def load_data():
    root = Path(__file__).parents[1]
    data_path = root / "data" / "patients.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Missing file: {data_path}")

    df = pd.read_csv(data_path)

    # Create AGE
    df['BIRTHDATE'] = pd.to_datetime(df['BIRTHDATE'], errors='coerce')
    df['AGE'] = (pd.Timestamp.today() - df['BIRTHDATE']).dt.days // 365

    # Fill missing values
    df['INCOME'] = df['INCOME'].fillna(0)
    df['HEALTHCARE_EXPENSES'] = df['HEALTHCARE_EXPENSES'].fillna(0)
    df['HEALTHCARE_COVERAGE'] = df['HEALTHCARE_COVERAGE'].fillna(0)

    df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce')
df['INCOME'] = pd.to_numeric(df['INCOME'], errors='coerce')
df['HEALTHCARE_EXPENSES'] = pd.to_numeric(df['HEALTHCARE_EXPENSES'], errors='coerce')
df['HEALTHCARE_COVERAGE'] = pd.to_numeric(df['HEALTHCARE_COVERAGE'], errors='coerce')

    # Aggregation
    report = df.groupby(['CITY', 'STATE', 'COUNTY']).agg({
        'HEALTHCARE_EXPENSES': 'mean',
        'HEALTHCARE_COVERAGE': 'mean',
        'INCOME': 'mean'
    }).reset_index()

    return df, report
