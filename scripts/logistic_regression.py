from model import Model
from data_processor import DataProcessor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


class LogReg(Model):
    def __init__(self, max_iter=10000, random_state=42):
        super().__init__(model_name="Logistic Regression", data_processor=DataProcessor(), random_state=random_state)
        self.max_iter = max_iter
        self.random_state = random_state
        [self.x_train, self.x_test, self.y_train, self.y_test] = self.split_data(test_size=0.2,
                                                                                 random_state=self.random_state)

    def train(self):
        """Train the logistic regression model."""
        scaler = StandardScaler()
        logreg = LogisticRegression(max_iter=10000, random_state=42)

        logreg.fit(self, self.x_train, self.y_train)
        self._set_model(logreg)

    def predict(self):
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(self.x_test)
