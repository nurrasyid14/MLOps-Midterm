import pandas as pd

from src.automl_pipeline import AutoMLPipeline


def main():
    # load data
    df = pd.read_excel("data/Concrete_Data.xls")

    # run AutoML
    pipeline = AutoMLPipeline(
        target_column="Concrete compressive strength(MPa, megapascals)" 
    )

    results = pipeline.run(df)

    print("\n=== AUTOML RESULTS ===")
    print(f"Best Model: {results['best_model']}")
    print("Metrics:", results["metrics"])


if __name__ == "__main__":
    main()
