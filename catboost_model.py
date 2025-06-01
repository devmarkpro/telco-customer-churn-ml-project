#!/usr/bin/env python3
"""
Train and predict with CatBoostClassifier on the Telco Customer Churn dataset.

Usage:
    conda activate csca-5622-supervised-learning
    python catboost_model.py
"""

import os
from catboost import CatBoostClassifier, Pool


def main():
    data_path = os.path.join("data", "Telco-Customer-Churn.csv")
    cd_path = "dataset_description.cd"

    data_pool = Pool(data=data_path, column_description=cd_path)
    model = CatBoostClassifier(random_seed=42, verbose=100)
    model.fit(data_pool)
    preds = model.predict(data_pool)
    print("First 10 predictions:", preds[:10])


if __name__ == "__main__":
    main()
