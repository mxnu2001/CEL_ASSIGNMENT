#!/usr/bin/env python3
"""Predict D180 LTV and ROAS for CSV rows using the saved product models."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "cel_d180_ltv.joblib"


def predict_dataframe(data: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Return predictions for user input rows using a loaded model artifact."""
    data = data.copy()
    if "cac" not in data.columns and "CAC" in data.columns:
        data = data.rename(columns={"CAC": "cac"})
    if data.empty:
        raise ValueError("Add at least one user row before requesting predictions.")
    if "product" not in data.columns:
        raise ValueError("Input CSV must contain a 'product' column.")
    if "cac" not in data.columns:
        raise ValueError("Input data must include 'cac' (acquisition cost per user) to calculate ROAS.")

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
        cac = pd.to_numeric(product_rows["cac"], errors="coerce")
        invalid_cac = cac.isna() | ~np.isfinite(cac) | (cac <= 0)
        if invalid_cac.any():
            row_labels = (product_rows.index[invalid_cac] + 2).tolist()
            raise ValueError(
                f"CAC must be a positive number for {product} row(s) {row_labels}."
            )
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
            "cac_inr": cac.to_numpy(),
            "predicted_d180_roas": prediction / cac.to_numpy(),
            "lower_80_roas": lower / cac.to_numpy(),
            "upper_80_roas": upper / cac.to_numpy(),
        }
        if "campaign" in product_rows.columns:
            result_columns["campaign"] = product_rows["campaign"].to_numpy()
        for identifier in ("test_user_id", "user_id"):
            if identifier in product_rows.columns:
                result_columns[identifier] = product_rows[identifier].to_numpy()
        block = pd.DataFrame(result_columns, index=product_rows.index)
        results.append(block)

    return pd.concat(results).sort_index()


def summarize_campaign_roas(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate predicted revenue and acquisition cost by product and campaign."""
    if "campaign" not in predictions.columns:
        return pd.DataFrame()
    rows = predictions.copy()
    rows["campaign"] = rows["campaign"].fillna("").astype(str).str.strip()
    rows = rows.loc[rows["campaign"] != ""]
    if rows.empty:
        return pd.DataFrame()
    summary = (
        rows.groupby(["product", "campaign"], as_index=False)
        .agg(
            users=("predicted_d180_ltv_inr", "size"),
            predicted_revenue_inr=("predicted_d180_ltv_inr", "sum"),
            lower_80_revenue_inr=("lower_80_inr", "sum"),
            upper_80_revenue_inr=("upper_80_inr", "sum"),
            acquisition_cost_inr=("cac_inr", "sum"),
        )
    )
    cost = summary["acquisition_cost_inr"]
    summary["predicted_d180_roas"] = summary["predicted_revenue_inr"] / cost
    summary["lower_80_roas"] = summary["lower_80_revenue_inr"] / cost
    summary["upper_80_roas"] = summary["upper_80_revenue_inr"] / cost
    return summary


def predict(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    selected_user_id: str | None = None,
) -> None:
    artifact = joblib.load(model_path)
    data = pd.read_csv(input_path)

    if selected_user_id is not None:
        identifier = "test_user_id" if "test_user_id" in data.columns else "user_id"
        if identifier not in data.columns:
            raise ValueError(
                "Single-user prediction needs a 'test_user_id' or 'user_id' column."
            )
        selected_rows = data[identifier].astype(str) == str(selected_user_id)
        if selected_rows.sum() != 1:
            raise ValueError(
                f"Expected exactly one row with {identifier}={selected_user_id!r}; "
                f"found {int(selected_rows.sum())}."
            )
        data = data.loc[selected_rows].copy()

    output = predict_dataframe(data, artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Wrote predictions to {output_path}")
    campaign_summary = summarize_campaign_roas(output)
    if not campaign_summary.empty:
        summary_path = output_path.with_name(f"{output_path.stem}_campaign_summary.csv")
        campaign_summary.to_csv(summary_path, index=False)
        print(f"Wrote campaign ROAS summary to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict D180 LTV and ROAS from Day 0–7 metrics and CAC."
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
    parser.add_argument(
        "--user-id", type=str, default=None,
        help="predict only the matching test_user_id (or user_id) from the input CSV",
    )
    args = parser.parse_args()
    predict(args.input_csv, args.output, args.model, args.user_id)


if __name__ == "__main__":
    main()
