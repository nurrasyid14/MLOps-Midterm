import pandas as pd

from src.pipeline import ManualPipeline


def main():
    # load data
    df = pd.read_excel("data/Concrete_Data.xls")

    # parameter tuning
    alphas = [0.01, 0.1, 1, 10, 100]

    # run pipeline
    pipeline = ManualPipeline(
        target_column="strength",  # sesuaikan nama kolom target
        alphas=alphas
    )

    results = pipeline.run(df)

    print("\n=== MANUAL RESULTS ===")
    print(f"Best Alpha: {results['best_alpha']}")
    print("Final Metrics:", results["final_metrics"])


if __name__ == "__main__":
    main()