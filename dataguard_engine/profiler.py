import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from .decorators import benchmark_and_log
from .exceptions import EmptyDatasetError, InvalidFileFormatError


class DatasetProfiler:
    """Production-grade Dataset Profiler using OOP and vectorized NumPy computations."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df: pd.DataFrame = self._load_data()
        self._validate_dataset()

    @benchmark_and_log("Dataset Ingestion")
    def _load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        ext = os.path.splitext(self.file_path)[-1].lower()
        if ext == ".csv":
            return pd.read_csv(self.file_path)
        elif ext in [".json", ".parquet"]:
            return pd.read_json(self.file_path) if ext == ".json" else pd.read_parquet(self.file_path)
        else:
            raise InvalidFileFormatError(f"Extension '{ext}' is not supported. Use CSV/JSON.", step="Ingestion")

    def _validate_dataset(self) -> None:
        if self.df.empty or len(self.df.columns) == 0:
            raise EmptyDatasetError("Dataset contains 0 rows or 0 columns.", step="Validation")

    # --- DUNDER METHODS ---
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, column_name: str) -> pd.Series:
        return self.df[column_name]

    def __repr__(self) -> str:
        return (
            f"DatasetProfiler(file='{os.path.basename(self.file_path)}', "
            f"rows={len(self.df)}, cols={len(self.df.columns)}, "
            f"ram_usage={self.memory_usage_mb:.3f} MB)"
        )

    def __str__(self) -> str:
        return f"Dataset Profile: {len(self.df)} Rows × {len(self.df.columns)} Columns | Memory: {self.memory_usage_mb:.2f} MB"

    # --- PROPERTIES ---
    @property
    def shape(self) -> tuple:
        return self.df.shape

    @property
    def memory_usage_mb(self) -> float:
        """Returns deep memory consumption in Megabytes."""
        return self.df.memory_usage(deep=True).sum() / (1024 * 1024)

    @property
    def numerical_columns(self) -> List[str]:
        return self.df.select_dtypes(include=[np.number]).columns.tolist()

    @property
    def categorical_columns(self) -> List[str]:
        return self.df.select_dtypes(exclude=[np.number]).columns.tolist()

    # --- NUMPY VECTORIZED STATISTICAL PROFILING ---
    @benchmark_and_log("Missing Values Audit")
    def get_missing_summary(self) -> Dict[str, Dict[str, Any]]:
        """Calculates missing values count and percentage per column."""
        total_rows = len(self.df)
        missing_data = {}
        for col in self.df.columns:
            null_count = int(self.df[col].isnull().sum())
            missing_data[col] = {
                "null_count": null_count,
                "null_percentage": (null_count / total_rows) * 100.0,
                "dtype": str(self.df[col].dtype),
            }
        return missing_data

    @benchmark_and_log("Vectorized Outlier Detection")
    def detect_outliers_numpy(self, column: str, method: str = "iqr") -> Dict[str, Any]:
        """Detects outliers using pure NumPy vectorization and boolean masking."""
        if column not in self.numerical_columns:
            raise ValueError(f"Column '{column}' is not a numeric column.")

        # Extract underlying NumPy array, dropping NaNs for statistical stability
        raw_values: np.ndarray = self.df[column].dropna().to_numpy()

        if len(raw_values) == 0:
            return {"outlier_count": 0, "indices": np.array([]), "thresholds": {}}

        if method == "iqr":
            q25, q75 = np.percentile(raw_values, 25), np.percentile(raw_values, 75)
            iqr = q75 - q25
            lower_bound = q25 - (1.5 * iqr)
            upper_bound = q75 + (1.5 * iqr)

            # Vectorized Boolean Mask
            outlier_mask = (raw_values < lower_bound) | (raw_values > upper_bound)
            outliers = raw_values[outlier_mask]

            return {
                "method": "IQR",
                "outlier_count": int(np.sum(outlier_mask)),
                "outlier_percentage": (np.sum(outlier_mask) / len(raw_values)) * 100.0,
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "sample_outliers": outliers[:5].tolist()
            }

        elif method == "z_score":
            mean = np.mean(raw_values)
            std = np.std(raw_values)
            if std == 0:
                return {"method": "Z-score", "outlier_count": 0, "outlier_percentage": 0.0}

            # Vectorized Z-Score calculation
            z_scores = np.abs((raw_values - mean) / std)
            outlier_mask = z_scores > 3.0
            outliers = raw_values[outlier_mask]

            return {
                "method": "Z-score",
                "outlier_count": int(np.sum(outlier_mask)),
                "outlier_percentage": (np.sum(outlier_mask) / len(raw_values)) * 100.0,
                "threshold": 3.0,
                "sample_outliers": outliers[:5].tolist()
            }
        else:
            raise ValueError("Method must be either 'iqr' or 'z_score'.")