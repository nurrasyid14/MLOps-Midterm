from .automl.functions import AutoMLFunctions
import os, json
from .prep.imputer import DataImputer
from .prep.norm import MinMaxNormalizer
from .logger import Logger
from src.evals.metrics import RegressionMetrics
from src.visuals.visualizer import Visualizer
from sklearn.model_selection import train_test_split


class AutoMLPipeline:
    def __init__(self, target_column):
        self.target_column = target_column

    def run(self, df):
        # 1. Impute
        imputer = DataImputer()
        df = imputer.fit_transform(df)

        # 2. Split dulu
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        # 3. Normalize (fit hanya train)
        X_train = train_df.drop(self.target_column, axis=1)
        y_train = train_df[self.target_column]

        X_test = test_df.drop(self.target_column, axis=1)
        y_test = test_df[self.target_column]

        normalizer = MinMaxNormalizer()
        X_train = normalizer.fit_transform(X_train)
        X_test = normalizer.transform(X_test)

        train_scaled = X_train.copy()
        train_scaled[self.target_column] = y_train

        test_scaled = X_test.copy()
        test_scaled[self.target_column] = y_test

        # 4. AutoML setup (pakai TRAIN saja)
        automl = AutoMLFunctions(self.target_column)
        automl.setup_env(train_scaled)

        # 5. Compare
        best_model, compare_results = automl.compare()

        # 6. Tune
        tuned_model, tune_results = automl.tune()

        # 7. Finalize
        final_model = automl.finalize()

        # 8. Predict (pakai TEST)
        preds = automl.predict(test_scaled)

        y_true = preds[self.target_column]
        y_pred = preds["prediction_label"]

        # 9. Metrics
        metrics = RegressionMetrics(y_true, y_pred)
        final_metrics = metrics.get_all_metrics()

        import datetime
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        os.makedirs("results/automl", exist_ok=True)

        # Visual
        viz = Visualizer(base_path="results/automl", run_id=run_id)
        viz.plot_pred_vs_actual(y_true, y_pred)
        viz.plot_residuals(y_true, y_pred)

        # Logging
        logger = Logger(filepath=f"results/automl/metrics_{run_id}.json")

        log_entry = {
            "run_id": run_id,
            "best_model": str(best_model),
            "metrics": final_metrics
        }

        logger.logs.append(log_entry)
        logger.save()

        # Save tables
        with open(f"results/automl/compare_{run_id}.json", "w") as f:
            json.dump(compare_results.to_dict(), f, indent=4)

        with open(f"results/automl/tune_{run_id}.json", "w") as f:
            json.dump(tune_results.to_dict(), f, indent=4)