import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, save_path="results/"):
        self.save_path = save_path

    def plot_pred_vs_actual(self, y_true, y_pred, filename="pred_vs_actual.png"):
        plt.figure()
        plt.scatter(y_true, y_pred)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Predicted vs Actual")
        plt.savefig(self.save_path + filename)
        plt.close()

    def plot_residuals(self, y_true, y_pred, filename="residuals.png"):
        residuals = y_true - y_pred
        plt.figure()
        plt.scatter(y_pred, residuals)
        plt.axhline(y=0)
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.savefig(self.save_path + filename)
        plt.close()

    def plot_alpha_vs_error(self, alphas, errors, filename="alpha_vs_error.png"):
        plt.figure()
        plt.plot(alphas, errors, marker='o')
        plt.xlabel("Alpha")
        plt.ylabel("Error (RMSE/MAE)")
        plt.title("Alpha vs Error")
        plt.savefig(self.save_path + filename)
        plt.close()