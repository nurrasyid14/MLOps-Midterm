from .automl.functions import AutoMLFunctions

from .prep.imputer import DataImputer
from src.prep.norm import MinMaxNormalizer

from src.evals.metrics import RegressionMetrics
from src.visuals.visualizer import Visualizer


class AutoMLPipeline:
    def __init__(self, target_column):
        self.target_column = target_column

    def run(self, df):
        # 1. Impute (biar fair dengan manual)
        imputer = DataImputer()
        df = imputer.fit_transform(df)

        # 2. Normalize (optional, PyCaret bisa handle, tapi kita samakan)
        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]

        normalizer = MinMaxNormalizer()
        X_scaled = normalizer.fit_transform(X)

        df_scaled = X_scaled.copy()
        df_scaled[self.target_column] = y

        # 3. AutoML setup
        automl = AutoMLFunctions(self.target_column)
        automl.setup_env(df_scaled)

        # 4. Compare models
        best_model, compare_results = automl.compare()

        # 5. Tune model
        tuned_model, tune_results = automl.tune()

        # 6. Finalize
        final_model = automl.finalize()

        # 7. Predict (on same data for simplicity)
        preds = automl.predict(df_scaled)

        y_true = preds[self.target_column]
        y_pred = preds["prediction_label"]

        # 8. Metrics
        metrics = RegressionMetrics(y_true, y_pred)
        final_metrics = metrics.get_all_metrics()

        # 9. Visualisasi
        viz = Visualizer()
        viz.plot_pred_vs_actual(y_true, y_pred)
        viz.plot_residuals(y_true, y_pred)

        return {
            "best_model": str(best_model),
            "metrics": final_metrics,
            "compare_table": compare_results.to_dict(),
            "tune_table": tune_results.to_dict()
        }