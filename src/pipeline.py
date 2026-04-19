from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np

from .prep.imputer import DataImputer
from .prep.norm import MinMaxNormalizer

from .manuals.regression import RidgeModel
from .tuner import ModelTuner as Tuner
from .manuals.regression.lgbm import LGBMModel

from .evals.metrics import RegressionMetrics
from .visuals.visualizer import Visualizer

from .logger.logger import Logger


class ManualPipeline:
    def __init__(self, target_column, alphas):
        self.target_column = target_column
        self.alphas = alphas

    def run(self, df):

        # 1. Split
        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 2. Impute
        imputer = DataImputer()
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # 3. Polynomial Features
        poly = PolynomialFeatures(degree=2, include_bias=False)

        X_train = poly.fit_transform(X_train)
        X_test = poly.transform(X_test)

        feature_names = poly.get_feature_names_out()

        X_train = pd.DataFrame(X_train, columns=feature_names, index=y_train.index)
        X_test = pd.DataFrame(X_test, columns=feature_names, index=y_test.index)

        # 4. Normalize
        normalizer = MinMaxNormalizer()
        X_train = normalizer.fit_transform(X_train)
        X_test = normalizer.transform(X_test)

        # 5. Tuning Loop

        import datetime
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = Logger(filepath=f"results/manuals/manuallogs_{run_id}.json")

        best_model = None
        best_config = None
        best_rmse = np.inf

        model_space = {
            "ridge": {
                "model": RidgeModel,
                "params": [{"alpha": alpha} for alpha in self.alphas]
            },
            "lgbm": {
                "model": LGBMModel,
                "params": [
                    {"n_estimators": 100, "learning_rate": 0.1},
                    {"n_estimators": 200, "learning_rate": 0.05}
                ]
            }
        }

        results_per_model = {}
        logs = []
        predictions = {}
        for model_name, config in model_space.items():
            ModelClass = config["model"]

            model_results = []
            best_rmse = np.inf
            best_model = None
            best_params = None

            for param in config["params"]:
                model = ModelClass(**param)
                model.train(X_train, y_train)
                y_pred = model.predict(X_test)

                metrics = RegressionMetrics(y_test, y_pred)
                result = metrics.get_all_metrics()

                log_entry = {
                    "model": model_name,
                    "params": param,
                    "RMSE": result["RMSE"],
                    "MAE": result["MAE"],
                    "R2": result["R2"]
                }

                logs.append(log_entry)
                model_results.append(log_entry)

                # best per model (bukan global)
                if result["RMSE"] < best_rmse:
                    best_rmse = result["RMSE"]
                    best_model = model
                    best_params = param

            # simpan hasil per model
            results_per_model[model_name] = {
                "best_params": best_params,
                "best_rmse": best_rmse,
                "results": model_results
            }

        # simpan log
        logger.logs = logs
        logger.save()

        # 2 prediksi 
        predictions = {}

        for model_name, info in results_per_model.items():
            ModelClass = model_space[model_name]["model"]

            model = ModelClass(**info["best_params"])
            model.train(X_train, y_train)

            y_pred = model.predict(X_test)

            metrics = RegressionMetrics(y_test, y_pred)

            predictions[model_name] = {
                "metrics": metrics.get_all_metrics(),
                "y_pred": y_pred
            }

        # 7. Visualisasi
        viz = Visualizer(base_path="results/manuals", run_id=run_id)

        for model_name, pred in predictions.items():
            viz.plot_pred_vs_actual(
                y_test,
                pred["y_pred"],
                filename=f"{model_name}_pred_vs_actual.png"
            )

            viz.plot_residuals(
                y_test,
                pred["y_pred"],
                filename=f"{model_name}_residuals.png"
            )

        # khusus ridge → alpha plot
        ridge_logs = [log for log in logs if log["model"] == "ridge"]

        if ridge_logs:
            alphas = [log["params"]["alpha"] for log in ridge_logs]
            rmse = [log["RMSE"] for log in ridge_logs]

            viz.plot_alpha_vs_error(
                alphas,
                rmse,
                filename="ridge_alpha_vs_error.png"
            )

        # 8. best_alpha Ridge
        best_alpha = None
        if "ridge" in results_per_model:
            best_alpha = results_per_model["ridge"]["best_params"].get("alpha")

        return {
            "models": results_per_model,
            "predictions": {
                k: v["metrics"] for k, v in predictions.items()
            },
            "best_alpha": best_alpha,
            "tuning_logs": logs
        }