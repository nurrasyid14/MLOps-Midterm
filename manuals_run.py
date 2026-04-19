import pandas as pd
import numpy as np
from src.pipeline import ManualPipeline


def main():
    # load data
    df = pd.read_excel("data/Concrete_Data.xls")

    # parameter tuning
    alphas = np.logspace(-4, 1, 20)

    # run pipeline
    pipeline = ManualPipeline(
        target_column="Concrete compressive strength(MPa, megapascals) ",
        alphas=alphas
    )

    results = pipeline.run(df)

    metrics = results["final_metrics"]

    print("\n=== MANUAL RESULTS ===")

    for model_name, metrics in results["predictions"].items():
        print(f"\nModel: {model_name}")
        print(f"RMSE: {metrics['RMSE']:.4f}")
        print(f"MAE : {metrics['MAE']:.4f}")
        print(f"R2  : {metrics['R2']:.4f}")

    if results["best_alpha"] is not None:
        print(f"\nBest Alpha (Ridge): {results['best_alpha']}")

    print("\n=== LEADERBOARD ===")

    sorted_models = sorted(
        results["predictions"].items(),
        key=lambda x: x[1]["RMSE"]
    )

    for i, (model, m) in enumerate(sorted_models, 1):
        print(f"{i}. {model} → RMSE: {m['RMSE']:.4f}")

if __name__ == "__main__":
    main()