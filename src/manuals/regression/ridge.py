from sklearn.linear_model import Ridge


class RidgeModel:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def set_alpha(self, alpha):
        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha)