#!/usr/bin/env python3
"""Predict D180 LTV for rows in a CSV using the saved product models."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "cel_d180_ltv.joblib"


def predict(input_path: Path, output_path: Path, model_path: Path) -> None:
    artifact = joblib.load(model_path)
    data = pd.read_csv(input_path)
    if "product" not in data.columns:
        raise ValueError("Input CSV must contain a 'product' column.")

    unknown = sorted(set(data["product"].dropna()) - set(artifact["products"]))
    if unknown:
        choices = ", ".join(artifact["products"])
        raise ValueError(f"Unknown product(s) {unknown}. Use: {choices}.")

    results = []
    for product, product_rows in data.groupby("product", sort=False, dropna=False):
        if pd.isna(product):
            raise ValueError("Every input row needs a product value.")
        details = artifact["products"][product]
        features = details["features"]
        missing = [column for column in features if column not in data.columns]
        if missing:
            raise ValueError(
                f"Input rows for {product} are missing feature columns: {missing}"
            )
        X = product_rows[features].apply(pd.to_numeric, errors="coerce")
        must_have = [name for name in features if name != "days_to_trial"]
        incomplete = X[must_have].isna().any(axis=1)
        if incomplete.any():
            source_rows = (product_rows.index[incomplete] + 2).tolist()
            raise ValueError(
                f"Missing or nonnumeric required feature values for {product} "
                f"on CSV row(s) {source_rows}. 'days_to_trial' may be blank when no trial started."
            )

        prediction = np.clip(details["model"].predict(X), 0, None)
        lower = np.clip(prediction + details["residual_p10"], 0, None)
        upper = np.clip(prediction + details["residual_p90"], 0, None)
        result_columns = {
            "product": product,
            "model": details["model_name"],
            "predicted_d180_ltv_inr": prediction,
            "lower_80_inr": lower,
            "upper_80_inr": upper,
        }
        for identifier in ("test_user_id", "user_id"):
            if identifier in product_rows.columns:
                result_columns[identifier] = product_rows[identifier].to_numpy()
        block = pd.DataFrame(result_columns, index=product_rows.index)
        results.append(block)

    output = pd.concat(results).sort_index()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Wrote predictions to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict D180 LTV from new Day 0–7 user metrics."
    )
    parser.add_argument("input_csv", type=Path, help="CSV containing new user metrics")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/predictions.csv"),
        help="where to write predictions (default: outputs/predictions.csv)",
    )
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL,
        help="saved model artifact (default: models/cel_d180_ltv.joblib)",
    )
    args = parser.parse_args()
    predict(args.input_csv, args.output, args.model)


if __name__ == "__main__":
    main()
