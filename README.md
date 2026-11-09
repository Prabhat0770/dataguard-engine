# 🛡️ DataGuard Engine

> Production-grade, vectorized data quality profiling and automated cleaning pipeline built with Python OOP, NumPy SIMD acceleration, and Pandas.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Managed with uv](https://img.shields.io/badge/managed%20by-uv-DE5FE9.svg)](https://github.com/astral-sh/uv)
[![Architecture](https://img.shields.io/badge/design-OOP%20%7C%20Vectorized-success.svg)](#architecture)

---

## 📌 Problem Statement
Data scientists often spend up to 60% of their workflow manually writing repetitive inspection scripts, parsing messy schemas, and fixing outliers using unvectorized Python loops that degrade system performance. 

**DataGuard Engine** provides a zero-overhead, modular profiling engine that:
1. Profiles incoming datasets using low-level NumPy boolean masks.
2. Applies strict Abstract Base Class (`ABC`) cleaning contracts.
3. Automatically downcasts memory representations (~50% RAM reduction).
4. Emits formatted executive Markdown reports and sub-millisecond audit traces.

---

## 🏗️ Architecture

```text
Raw Ingestion (CSV / JSON)
        │
        ▼
DatasetProfiler (Dunders: __len__, __getitem__, __repr__)
        │
        ├──> Vectorized Outlier Detection (NumPy SIMD IQR / Z-Score)
        └──> Schema & Memory Profiling (@property)
        │
        ▼
DataCleanerPipeline (Abstract Base Class Contract: BaseTransformer)
        ├──> MissingValueImputer (Mode / Median Strategy)
        ├──> VectorizedOutlierCapper (NumPy np.clip boundary capping)
        └──> MemoryOptimizer (Bit-width downcasting & category coercion)
        │
        ▼
Output Artifacts
        ├──> Cleaned CSV Export (data/cleaned_transactions.csv)
        ├──> Audit Traces (logs/pipeline.log via @benchmark_and_log)
        └──> Markdown Health Report (reports/data_health_report.md)
```

---

## ⚡ Performance & Optimization Benchmarks

Benchmarks recorded on `sample_transactions.csv` (1,000 rows):

| Pipeline Step | Mechanism | Latency | Status |
| :--- | :--- | :--- | :--- |
| **Ingestion** | `pd.read_csv` with path validation | ~3.9 ms | `SUCCESS` |
| **Pipeline Cleaning** | Vectorized Capping + Downcasting | ~11.0 ms | `SUCCESS` |
| **Outlier Detection** | Pure NumPy Vectorized Masking | ~1.5 - 4.5 ms | `SUCCESS` |
| **Markdown Reporter** | Direct disk stream context manager | ~41.5 ms | `SUCCESS` |

### 💾 Memory Footprint Reduction
- **Initial RAM Footprint:** `0.128 MB`
- **Post-Optimization Footprint:** `0.064 MB`
- **Total RAM Saved:** **`~49.75%`** via type downcasting and low-cardinality categorization.

---

## 🛠️ Concepts Demonstrated
- **Runtime & Packaging:** Fast dependency resolution via `uv` package manager (`pyproject.toml`).
- **Object-Oriented Design:** Abstract Base Classes (`ABC`, `@abstractmethod`), Method Overriding, Encapsulation, Polymorphism.
- **Python Mechanics:** Parameterized Decorators with `*args` / `**kwargs`, Custom Exception Hierarchies (`raise DataGuardError`), Dunder methods (`__len__`, `__getitem__`, `__repr__`, `__str__`).
- **High-Performance Computation:** NumPy array indexing, IQR percentiles, SIMD vectorization without explicit loops.
- **Audit & Logging:** High-precision execution tracking via `time.perf_counter()` and context-managed file logging.

---

## 🚀 Quickstart

### Prerequisites
Install [uv](https://github.com/astral-sh/uv):
```powershell
powershell -ExecutionPolicy ByPass -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"
```

### Installation & Execution
```bash
# Clone the repository
git clone [https://github.com/Prabhat0770/dataguard-engine.git](https://github.com/Prabhat0770/dataguard-engine.git)
cd dataguard-engine

# Sync dependencies and virtual environment
uv sync

# Generate sample transactions dataset
uv run data/generate_sample.py

# Run the complete end-to-end engine
uv run main.py
```