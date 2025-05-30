import pandas as pd

class Data:
    def __init__(self):
        self.data_path = "../data/Telco-Customer-Churn.csv"
        self.data = None
        self.demographic_features = ["gender", "SeniorCitizen", "Partner", "Dependents"]
        self.service_features = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                            "OnlineBackup", "DeviceProtection", "StreamingTV", "StreamingMovies", "TechSupport"]
        self.payment_features = ["Contract", "PaperlessBilling", "PaymentMethod"]
        self.binary_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'SeniorCitizen']
        self.categorical_features = list(set(self.service_features + self.payment_features) - set(self.binary_features))
        self.numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

    def load_data(self) -> pd.DataFrame:
        """Load data from the specified path."""
        try:
            self.data = pd.read_csv(self.data_path)
            return self.data
        except Exception as e:
            raise print(f"Error loading data: path:{self.data_path}, error: {e}")

    def get_data(self) -> pd.DataFrame:
        """Return the loaded data."""
        return self.data