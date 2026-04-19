import matplotlib.pyplot as plt
import os
import datetime


class Visualizer:
    def __init__(self, base_path="results/manuals", run_id=None):
        self.base_path = base_path

        # jika run_id tidak diberikan → generate otomatis
        if run_id is None:
            self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.run_id = run_id

        # buat folder jika belum ada
        os.makedirs(self.base_path, exist_ok=True)

    def _build_path(self, filename):
        filename = f"{self.run_id}_{filename}"
        return os.path.join(self.base_path, filename)

    def plot_pred_vs_actual(self, y_true, y_pred, filename="pred_vs_actual.png"):
        plt.figure()
        plt.scatter(y_true, y_pred)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Predicted vs Actual")

        plt.savefig(self._build_path(filename))
        plt.close()

    def plot_residuals(self, y_true, y_pred, filename="residuals.png"):
        residuals = y_true - y_pred
        plt.figure()
        plt.scatter(y_pred, residuals)
        plt.axhline(y=0)
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")

        plt.savefig(self._build_path(filename))
        plt.close()

    def plot_alpha_vs_error(self, alphas, errors, filename="alpha_vs_error.png"):
        plt.figure()
        plt.plot(alphas, errors, marker='o')
        plt.xlabel("Alpha")
        plt.ylabel("Error (RMSE/MAE)")
        plt.title("Alpha vs Error")

        plt.savefig(self._build_path(filename))
        plt.close()