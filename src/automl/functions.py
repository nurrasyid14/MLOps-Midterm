from pycaret.regression import (
    setup,
    compare_models,
    tune_model,
    finalize_model,
    predict_model,
    pull
)


class AutoMLFunctions:
    def __init__(self, target_column, session_id=42):
        self.target_column = target_column
        self.session_id = session_id
        self.best_model = None
        self.tuned_model = None
        self.final_model = None

    def setup_env(self, df):
        setup(
            data=df,
            target=self.target_column,
            session_id=self.session_id,
            verbose=False
        )

    def compare(self):
        self.best_model = compare_models()
        results = pull()
        return self.best_model, results

    def tune(self):
        self.tuned_model = tune_model(self.best_model)
        results = pull()
        return self.tuned_model, results

    def finalize(self):
        self.final_model = finalize_model(self.tuned_model)
        return self.final_model

    def predict(self, df):
        preds = predict_model(self.final_model, data=df)
        return preds