#!/usr/bin/env python3
"""Train the already-selected product models and save a reusable artifact."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "synthetic_users.csv"
SELECTION_PATH = ROOT / "outputs" / "model_selection_cv_metrics.csv"
ARTIFACT_PATH = ROOT / "models" / "cel_d180_ltv.joblib"
TARGET = "d180_ltv"

COMMON_FEATURES = [
    "active_days_7",
    "sessions_7",
    "avg_session_min_7",
    "sessions_per_active_day_7",
    "d1_retained",
    "d3_retained",
    "d7_retained",
    "days_since_last_active_7",
    "session_momentum_7",
    "median_gap_hours_7",
    "social_actions_7",
]
PRODUCT_FEATURES = {
    "subscription": [
        "trial_started_d7",
        "subscription_started_d7",
        "days_to_trial",
    ],
    "ad_supported": [
        "ad_impressions_7",
        "ad_views_7",
        "rewarded_ads_watched_7",
        "ad_view_rate_7",
        "rewarded_ad_rate_7",
    ],
}
ESTIMATORS = {
    "linear_regression": LinearRegression,
    "ridge_regression": lambda: Ridge(alpha=1.0),
}


def make_estimator(model_name: str):
    try:
        estimator = ESTIMATORS[model_name]()
    except KeyError as exc:
        raise ValueError(f"Unsupported selected model: {model_name}") from exc
    return make_pipeline(SimpleImputer(strategy="median"), estimator)


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    selection = pd.read_csv(SELECTION_PATH)
    if TARGET not in data:
        raise ValueError(f"Training data is missing target column {TARGET!r}.")

    products: dict[str, dict] = {}
    for product, product_features in PRODUCT_FEATURES.items():
        rows = selection.loc[selection["product"] == product]
        if rows.empty:
            raise ValueError(f"No recorded model selection found for {product}.")
        selected = rows.loc[rows["CV_MAE"].idxmin()]
        model_name = str(selected["model"])
        if model_name not in ESTIMATORS:
            raise ValueError(
                f"Selected model {model_name!r} for {product} is not supported by this trainer."
            )

        subset = data.loc[data["product"] == product]
        features = COMMON_FEATURES + product_features
        missing = sorted(set(features + [TARGET]) - set(subset.columns))
        if missing:
            raise ValueError(f"Training data for {product} is missing columns: {missing}")
        X = subset[features]
        y = subset[TARGET]

        # Estimate prediction-range offsets from out-of-fold residuals, then fit
        # the final model on every labeled row for this product.
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_predictions = cross_val_predict(make_estimator(model_name), X, y, cv=cv)
        residuals = y.to_numpy() - oof_predictions
        residual_p10, residual_p90 = np.quantile(residuals, [0.10, 0.90])
        model = make_estimator(model_name).fit(X, y)

        products[product] = {
            "model": model,
            "model_name": model_name,
            "features": features,
            "training_rows": int(len(subset)),
            "selection_cv_mae": float(selected["CV_MAE"]),
            "residual_p10": float(residual_p10),
            "residual_p90": float(residual_p90),
        }

    artifact = {
        "format_version": 1,
        "target": TARGET,
        "currency": "INR",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "products": products,
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT_PATH)
    print(f"Saved model artifact: {ARTIFACT_PATH.relative_to(ROOT)}")
    for product, details in products.items():
        print(
            f"{product}: {details['model_name']} "
            f"(trained on {details['training_rows']:,} rows; "
            f"selection CV MAE ₹{details['selection_cv_mae']:.2f})"
        )


if __name__ == "__main__":
    main()
