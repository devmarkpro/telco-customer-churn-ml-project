from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


class Model:
    def __init__(self, model_name="", data_processor=None, random_state=42):
        if data_processor is None:
            raise ValueError("Data processor must be provided.")
        if not model_name or model_name.strip():
            raise ValueError("Model name must be provided.")

        self.data_processor = data_processor
        self.random_state = random_state

        self.df = self.data_processor.process_data()
        self.xy = [self.df.drop(columns=['Churn']), self.df['Churn']]
        self.model_name = model_name
        self.model = None

    def split_data(self, test_size=0.2, random_state=None):
        """Split the data into training and testing sets."""

        if random_state is None:
            random_state = self.random_state
        [x, y] = self.xy
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        return x_train, x_test, y_train, y_test

    def plot_roc_curve(self, X_test, y_test):
        # Try getting scores from predict_proba or decision_function
        if hasattr(self.model, "predict_proba"):
            y_scores = self.model.predict_proba(X_test)[:, 1]
        elif hasattr(self.model, "decision_function"):
            y_scores = self.model.decision_function(X_test)
        else:
            raise ValueError(f"{self.model_name} does not support ROC curve plotting.")

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)

        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{self.model_name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def _set_model(self, model, name=None):
        """Set the model."""
        if model is None:
            raise ValueError("Model cannot be None.")

        if name is not None:
            name = model.__class__.__name__

        self.model_name = name
