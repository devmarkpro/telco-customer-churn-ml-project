from data import Data
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataProcessor(Data):
    def __init__(self):
        super().__init__()

    def process_data(self) -> pd.DataFrame:
        """Process the loaded data."""
        if self.data is None:
            raise ValueError("Data not loaded. Please load the data first.")

        res = {}
        df = self._copy(self.data)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

        self._map_binary_features(df, inplace=True)
        self._drop_extra_features(df, inplace=True)

        return df

    def _map_binary_features(self, df: pd.DataFrame, inplace=False) -> pd.DataFrame:
        # Mapping 'Yes'/'No' to 1/0
        if not inplace:
            df = self._copy(df)
        df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
        df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
        df['PhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0})
        df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

        # Mapping 'Male'/'Female' to 1/0
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

        return df

    def _drop_extra_features(self, df: pd.DataFrame, inplace=False) -> pd.DataFrame:
        """Drop features that are not needed for analysis."""
        if not inplace:
            df = self._copy(df)
        # Drop 'customerID' as it is not useful for analysis
        df.drop(columns=['customerID'], inplace=True)
        return df

    def _copy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a deep copy of the DataFrame."""
        if df is None:
            return self.data.copy(deep=True)
        return df.copy(deep=True)

    def _scale_numeric_features(self, df: pd.DataFrame, inplace=False) -> pd.DataFrame:
        """Scale numeric features to a range of 0 to 1."""
        scaler = StandardScaler()
        if not inplace:
            df = self._copy(df)
        numeric_features = self.numeric_features
        for feature in numeric_features:
            df[feature] = scaler.fit_transform(df[['tenure', 'MonthlyCharges', 'TotalCharges']])
        return df

    def get_processed_data(self) -> pd.DataFrame:
        """Return the processed data."""
        return self.process_data()
