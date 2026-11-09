from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import pandas as pd

from .decorators import benchmark_and_log


class BaseTransformer(ABC):
    """Abstract Base Class enforcing a strict contract for all data cleaning steps."""

    @abstractmethod
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mandatory method that every child transformer must implement."""
        pass


class MissingValueImputer(BaseTransformer):
    """Handles missing values: Median for numerical features, Mode for categorical."""

    def __init__(self, strategy_num: str = "median", fill_cat_default: str = "Unknown"):
        self.strategy_num = strategy_num
        self.fill_cat_default = fill_cat_default

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        
        # 1. Numerical Imputation
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df_clean[col].isnull().any():
                val = df_clean[col].median() if self.strategy_num == "median" else df_clean[col].mean()
                df_clean[col] = df_clean[col].fillna(val)

        # 2. Categorical Imputation
        cat_cols = df_clean.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            if df_clean[col].isnull().any():
                mode_series = df_clean[col].mode()
                fill_val = mode_series[0] if not mode_series.empty else self.fill_cat_default
                df_clean[col] = df_clean[col].fillna(fill_val)

        return df_clean


class VectorizedOutlierCapper(BaseTransformer):
    """Caps outliers at IQR boundaries using pure NumPy vectorized clipping (Winsorization)."""

    def __init__(self, columns: Optional[List[str]] = None, factor: float = 1.5):
        self.columns = columns
        self.factor = factor

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        target_cols = self.columns or df_clean.select_dtypes(include=[np.number]).columns.tolist()

        for col in target_cols:
            if col not in df_clean.columns:
                continue

            raw_arr = df_clean[col].to_numpy()
            q25, q75 = np.percentile(raw_arr[~np.isnan(raw_arr)], 25), np.percentile(raw_arr[~np.isnan(raw_arr)], 75)
            iqr = q75 - q25
            lower_bound = q25 - (self.factor * iqr)
            upper_bound = q75 + (self.factor * iqr)

            # Vectorized in-place clipping via NumPy
            df_clean[col] = np.clip(raw_arr, lower_bound, upper_bound)

        return df_clean


class MemoryOptimizer(BaseTransformer):
    """Downcasts float/int bit widths and converts low-cardinality objects to categorical."""

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()

        # 1. Downcast numerical datatypes
        for col in df_clean.select_dtypes(include=["float64"]).columns:
            df_clean[col] = pd.to_numeric(df_clean[col], downcast="float")

        for col in df_clean.select_dtypes(include=["int64"]).columns:
            df_clean[col] = pd.to_numeric(df_clean[col], downcast="integer")

        # 2. Convert low cardinality strings to category (saves heavy RAM)
        for col in df_clean.select_dtypes(include=["object"]).columns:
            num_unique = df_clean[col].nunique()
            total_rows = len(df_clean)
            if num_unique / total_rows < 0.2:  # Less than 20% unique values
                df_clean[col] = df_clean[col].astype("category")

        return df_clean


class DataCleanerPipeline:
    """Orchestrates sequential execution of BaseTransformer subclasses."""

    def __init__(self):
        self.transformers: List[BaseTransformer] = []

    def add_step(self, transformer: BaseTransformer) -> "DataCleanerPipeline":
        if not isinstance(transformer, BaseTransformer):
            raise TypeError(f"{type(transformer).__name__} must inherit from BaseTransformer")
        self.transformers.append(transformer)
        return self

    @benchmark_and_log("Pipeline Data Cleaning")
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        current_df = df
        for step in self.transformers:
            current_df = step.fit_transform(current_df)
        return current_df