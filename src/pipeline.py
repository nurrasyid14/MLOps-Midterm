from sklearn.model_selection import train_test_split

from .prep.imputer import DataImputer
from .prep.norm import MinMaxNormalizer

from .manuals.regression import RidgeModel
from .tuner import RidgeTuner

from .evals.metrics import RegressionMetrics
from .visuals.visualizer import Visualizer


class ManualPipeline:
    def __init__(self, target_column, alphas):
        self.target_column = target_column
        self.alphas = alphas

    def run(self, df):
        # 1. Impute
        imputer = DataImputer()
        df = imputer.fit_transform(df)

        # 2. Split X y
        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]

        # 3. Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 4. Normalize
        normalizer = MinMaxNormalizer()
        X_train = normalizer.fit_transform(X_train)
        X_test = normalizer.transform(X_test)

        # 5. Tuning
        model = RidgeModel()
        tuner = RidgeTuner(model, self.alphas)

        best_alpha, best_metrics = tuner.tune(
            X_train, X_test, y_train, y_test
        )

        # 6. Train best model again
        best_model = RidgeModel(alpha=best_alpha)
        best_model.train(X_train, y_train)
        y_pred = best_model.predict(X_test)

        # 7. Final metrics
        metrics = RegressionMetrics(y_test, y_pred)
        final_metrics = metrics.get_all_metrics()

        # 8. Visualizations
        viz = Visualizer()

        viz.plot_pred_vs_actual(y_test, y_pred)
        viz.plot_residuals(y_test, y_pred)

        # ambil logs dari tuner
        logs = tuner.logger.get_logs()
        alphas = [log["alpha"] for log in logs]
        rmse = [log["RMSE"] for log in logs]

        viz.plot_alpha_vs_error(alphas, rmse)

        return {
            "best_alpha": best_alpha,
            "tuning_metrics": best_metrics,
            "final_metrics": final_metrics
        }