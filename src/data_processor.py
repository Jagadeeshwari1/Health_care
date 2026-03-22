import pandas as pd
import glob
from pathlib import Path

def load_and_merge_data():
    root_path = Path(__file__).parents[1]
    data_dir = root_path / "data"

    # Load Patients
    patients = pd.read_csv(data_dir / "patients.csv")

    # Load Encounters
    search_pattern = str(data_dir / "encounters_part_*.csv")
    encounter_files = glob.glob(search_pattern)
    encounters = pd.concat((pd.read_csv(f) for f in encounter_files), ignore_index=True)

    # Clean Financials
    for col in ['INCOME', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']:
        if col in patients.columns:
            patients[col] = pd.to_numeric(
                patients[col].astype(str).str.replace(r'[\$,]', '', regex=True), 
                errors='coerce'
            ).fillna(0)

    # Create Tiers
    patients['INCOME_TIER'] = patients['INCOME'].apply(
        lambda x: 'Low Income' if x < 35000 else ('Middle Income' if x < 85000 else 'High Income')
    )

    # Merge
    merged_data = pd.merge(encounters, patients, left_on='PATIENT', right_on='Id')
    
    # Aggregated Report
    report = merged_data.groupby(['CITY', 'INCOME_TIER']).agg({
        'TOTAL_CLAIM_COST': 'mean',
        'HEALTHCARE_EXPENSES': 'mean'
    }).reset_index()

    return merged_data, report
