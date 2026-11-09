import os
from dataguard_engine import (
    DatasetProfiler,
    DataCleanerPipeline,
    MissingValueImputer,
    VectorizedOutlierCapper,
    MemoryOptimizer,
)


def run_pipeline():
    raw_path = "data/sample_transactions.csv"
    clean_path = "data/cleaned_transactions.csv"

    print("\n" + "=" * 60)
    print("🛡️  DATAGUARD: END-TO-END PROFILING & CLEANING PIPELINE")
    print("=" * 60)

    # 1. Ingestion & Profiling
    profiler = DatasetProfiler(raw_path)
    initial_memory = profiler.memory_usage_mb
    print(f"📦 Initial Dataset: {len(profiler)} rows | RAM: {initial_memory:.3f} MB")

    # 2. Build Cleaning Pipeline using Polymorphism & ABC
    pipeline = DataCleanerPipeline()
    pipeline.add_step(MissingValueImputer(strategy_num="median"))
    pipeline.add_step(VectorizedOutlierCapper(columns=["amount"], factor=1.5))
    pipeline.add_step(MemoryOptimizer())

    # 3. Execute Pipeline
    cleaned_df = pipeline.run(profiler.df)

    # 4. Save Cleaned Dataset
    cleaned_df.to_csv(clean_path, index=False)
    print(f"💾 Cleaned dataset saved to: {clean_path}")

    # 5. Measure Optimization & Integrity Results
    final_memory = cleaned_df.memory_usage(deep=True).sum() / (1024 * 1024)
    ram_reduction = ((initial_memory - final_memory) / initial_memory) * 100

    print("\n📈 PIPELINE OPTIMIZATION AUDIT:")
    print(f"  • Missing Values Left : {int(cleaned_df.isnull().sum().sum())}")
    print(f"  • Max 'amount' Before  : {profiler.df['amount'].max():.2f}")
    print(f"  • Max 'amount' After   : {cleaned_df['amount'].max():.2f} (Capped cleanly)")
    print(f"  • Initial RAM Footprint: {initial_memory:.3f} MB")
    print(f"  • Optimized Footprint  : {final_memory:.3f} MB")
    print(f"  • Total RAM Reduction  : {ram_reduction:.2f}%\n")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_pipeline()  