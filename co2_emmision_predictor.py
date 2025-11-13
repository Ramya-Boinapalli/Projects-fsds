# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 18:36:05 2025

@author: Ramya
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


def train_and_export(
    data_path=r"C:\Users\Ramya\Downloads\archive (1)\co2_emissions.csv",
    output_dir="models"
):
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    df = pd.read_csv(data_path)
    print(f"✅ Dataset loaded successfully with shape: {df.shape}")
    print("Columns:", df.columns.tolist())

    # Ensure required columns exist
    expected_cols = ["year", "month", "decimal_date", "co2_ppm"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    # Drop missing values
    df = df.dropna(subset=expected_cols)

    # Prepare features and target
    X = df[["year", "month", "decimal_date"]]
    y = df["co2_ppm"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    # Hyperparameter tuning
    param_grid = {
        "rf__n_estimators": [100, 200],
        "rf__max_depth": [None, 10, 20]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="neg_mean_absolute_error", verbose=1)
    grid.fit(X_train, y_train)

    print("✅ Best params:", grid.best_params_)
    best_model = grid.best_estimator_

    # Evaluate model
    preds = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)

    print(f"📊 Test MAE: {mae:.3f}")
    print(f"📊 Test RMSE: {rmse:.3f}")
    print(f"📊 Test R²: {r2:.3f}")

    # Save model
    model_path = os.path.join(output_dir, "rf_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"💾 Model saved to: {model_path}")


if __name__ == "__main__":
    train_and_export()
