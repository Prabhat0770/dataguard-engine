import numpy as np
import pandas as pd
import os

def create_sample_data(filepath: str = "data/sample_transactions.csv"):
    np.random.seed(42)
    n_rows = 1000

    # 1. Base clean features
    transaction_ids = [f"TXN_{1000 + i}" for i in range(n_rows)]
    ages = np.random.randint(18, 70, size=n_rows).astype(float)
    amounts = np.random.normal(loc=150.0, scale=40.0, size=n_rows)
    cities = np.random.choice(["Delhi", "Mumbai", "Bangalore", "Noida", None], size=n_rows, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    is_fraud = np.random.choice([0, 1], size=n_rows, p=[0.96, 0.04])

    # 2. Inject dirty values & missing data
    missing_indices = np.random.choice(n_rows, size=50, replace=False)
    ages[missing_indices] = np.nan

    # 3. Inject extreme outliers in amounts
    outlier_indices = np.random.choice(n_rows, size=15, replace=False)
    amounts[outlier_indices] = amounts[outlier_indices] * 12.0  # Massive spikes

    df = pd.DataFrame({
        "transaction_id": transaction_ids,
        "age": ages,
        "amount": amounts,
        "city": cities,
        "is_fraud": is_fraud
    })

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Sample dataset created at: {filepath} ({len(df)} rows)")

if __name__ == "__main__":
    create_sample_data()