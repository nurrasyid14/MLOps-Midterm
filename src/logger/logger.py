import json
import os


class Logger:
    def __init__(self, filepath="results/logs.json"):
        self.filepath = filepath
        self.logs = []

        # pastikan folder ada
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def log(self, alpha, metrics_dict):
        entry = {
            "alpha": alpha,
            "RMSE": metrics_dict.get("RMSE"),
            "MAE": metrics_dict.get("MAE"),
            "R2": metrics_dict.get("R2")
        }
        self.logs.append(entry)

    def save(self):
        with open(self.filepath, "w") as f:
            json.dump(self.logs, f, indent=4)

    def get_logs(self):
        return self.logs