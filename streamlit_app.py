#!/usr/bin/env python3
"""Manual, row-based Streamlit interface for the saved D180 LTV models."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from predict_ltv import predict_dataframe


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "cel_d180_ltv.joblib"

FIELD_DETAILS = {
    "active_days_7": (
        "Active days (0–7)",
        "Distinct days the user was active during Days 0–7. Integer from 0 to 7.",
        0,
        7,
        1,
    ),
    "sessions_7": (
        "Sessions (Days 0–7)",
        "Total sessions started by the user during Days 0–7.",
        0,
        None,
        1,
    ),
    "avg_session_min_7": (
        "Average session duration (minutes)",
        "Average duration of the user's sessions during the first seven days.",
        0,
        None,
        0.1,
    ),
    "sessions_per_active_day_7": (
        "Sessions per active day",
        "Usage frequency conditional on active days; sessions_7 divided by active_days_7.",
        0,
        None,
        0.1,
    ),
    "d1_retained": (
        "Day 1 retained (0/1)",
        "Whether the user returned on Day 1 after installation. Enter 1 for yes, 0 for no.",
        0,
        1,
        1,
    ),
    "d3_retained": (
        "Day 3 retained (0/1)",
        "Whether the user returned on Day 3. Enter 1 for yes, 0 for no.",
        0,
        1,
        1,
    ),
    "d7_retained": (
        "Day 7 retained (0/1)",
        "Whether the user met the Day 7 retention definition. Enter 1 for yes, 0 for no.",
        0,
        1,
        1,
    ),
    "days_since_last_active_7": (
        "Days since last activity (as of Day 7)",
        "Recency of the user's last activity in the Day 0–7 window. Integer from 0 to 7.",
        0,
        7,
        1,
    ),
    "session_momentum_7": (
        "Session momentum",
        "Whether activity rose or fell toward the end of the observation window; late-period sessions divided by early-period sessions.",
        0,
        None,
        0.1,
    ),
    "median_gap_hours_7": (
        "Median session gap (hours)",
        "Typical time between consecutive sessions. Lower values mean more frequent returns.",
        0,
        None,
        0.1,
    ),
    "social_actions_7": (
        "Social actions",
        "Count of social, referral, or invite actions during the first seven days.",
        0,
        None,
        1,
    ),
    "trial_started_d7": (
        "Trial started by Day 7 (0/1)",
        "Whether a subscription user started a trial within the first seven days.",
        0,
        1,
        1,
    ),
    "subscription_started_d7": (
        "Paid subscription by Day 7 (0/1)",
        "Whether the user converted to a paid subscription within the first seven days.",
        0,
        1,
        1,
    ),
    "days_to_trial": (
        "Days to trial",
        "Days from install to trial start (1–7). Leave blank if no trial was started.",
        1,
        7,
        1,
    ),
    "ad_impressions_7": (
        "Ad impressions (Days 0–7)",
        "Number of ads served or shown to the user during Days 0–7.",
        0,
        None,
        1,
    ),
    "ad_views_7": (
        "Qualifying ad views (Days 0–7)",
        "Number of completed or otherwise qualifying ad views during Days 0–7.",
        0,
        None,
        1,
    ),
    "rewarded_ads_watched_7": (
        "Rewarded ads watched",
        "Number of rewarded ads watched during the first seven days.",
        0,
        None,
        1,
    ),
    "ad_view_rate_7": (
        "Ad view rate",
        "Qualifying ad views divided by ad impressions. Enter as a ratio, such as 0.5 for 50%.",
        0,
        None,
        0.01,
    ),
    "rewarded_ad_rate_7": (
        "Rewarded ad rate",
        "Rewarded ads watched divided by ad impressions. Enter as a ratio, such as 0.2 for 20%.",
        0,
        None,
        0.01,
    ),
}


def main() -> None:
    st.set_page_config(page_title="D180 LTV Predictor", layout="wide")
    st.title("D180 LTV Predictor")
    st.write(
        "Enter Day 0–7 user metrics below. Add one row per user, choose the product, "
        "and select **Predict D180 LTV** to see the estimate and its empirical 80% range."
    )
    st.caption(
        "The field help text follows the CEL LTV Feature Dictionary. Values must describe "
        "the first seven days after install."
    )

    if not MODEL_PATH.exists():
        st.error("Saved model not found. From the project folder, run `python train_model.py` first.")
        st.stop()
    artifact = joblib.load(MODEL_PATH)

    common_features = [
        feature
        for feature in artifact["products"]["subscription"]["features"]
        if feature not in {"trial_started_d7", "subscription_started_d7", "days_to_trial"}
    ]
    all_features = common_features + [
        "trial_started_d7",
        "subscription_started_d7",
        "days_to_trial",
        "ad_impressions_7",
        "ad_views_7",
        "rewarded_ads_watched_7",
        "ad_view_rate_7",
        "rewarded_ad_rate_7",
    ]
    input_columns = ["test_user_id", "product"] + all_features
    if "ltv_placeholder_row" not in st.session_state:
        synthetic_data = pd.read_csv(ROOT / "data" / "synthetic_users.csv")
        example_product = synthetic_data["product"].sample(n=1).iloc[0]
        example = synthetic_data.loc[
            synthetic_data["product"] == example_product
        ].sample(n=1).iloc[0]
        placeholder_row = {column: None for column in input_columns}
        placeholder_row["test_user_id"] = "Example-001"
        placeholder_row["product"] = example_product
        active_features = set(artifact["products"][example_product]["features"])
        for feature in all_features:
            if (
                feature in active_features
                and feature in example.index
                and pd.notna(example[feature])
            ):
                placeholder_row[feature] = example[feature]
        st.session_state["ltv_placeholder_row"] = placeholder_row
    initial_rows = pd.DataFrame(
        [st.session_state["ltv_placeholder_row"]], columns=input_columns
    )

    column_config = {
        "test_user_id": st.column_config.TextColumn(
            "User ID", help="Unique label used to match each output to its input row."
        ),
        "product": st.column_config.SelectboxColumn(
            "Product",
            help="Choose the model that matches the user's product.",
            options=["subscription", "ad_supported"],
        ),
    }
    for feature in all_features:
        label, description, minimum, maximum, step = FIELD_DETAILS[feature]
        column_config[feature] = st.column_config.NumberColumn(
            label,
            help=description,
            min_value=minimum,
            max_value=maximum,
            step=step,
            format="%d" if step == 1 else "%.2f",
        )

    with st.expander("Feature descriptions", expanded=False):
        description_rows = [
            {"Input": FIELD_DETAILS[name][0], "Description": FIELD_DETAILS[name][1]}
            for name in all_features
        ]
        st.dataframe(pd.DataFrame(description_rows), hide_index=True, use_container_width=True)
        st.caption(
            "Campaign, CAC, actual D180 LTV, and derived fields not used by the selected "
            "models are excluded from the prediction inputs."
        )

    st.info(
        "The first row is prefilled with a randomly sampled synthetic example. "
        "Replace these example values with the user's Day 0–7 metrics."
    )
    st.subheader("User inputs")
    st.write(
        "Edit the first row or use the table's add-row control to enter more users. "
        "Hover over a column heading for its short description. Leave the unused "
        "product-specific fields blank."
    )
    entered = st.data_editor(
        initial_rows,
        num_rows="dynamic",
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        key="ltv_user_inputs",
    )

    if st.button("Predict D180 LTV", type="primary"):
        try:
            inputs = entered.copy()
            inputs["test_user_id"] = inputs["test_user_id"].fillna("").astype(str).str.strip()
            if inputs["test_user_id"].eq("").any():
                raise ValueError("Enter a unique User ID for every row.")
            if inputs["test_user_id"].duplicated().any():
                raise ValueError("Each User ID must be unique.")
            predictions = predict_dataframe(inputs, artifact)
            st.subheader("Predicted D180 LTV")
            st.dataframe(
                predictions,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "predicted_d180_ltv_inr": st.column_config.NumberColumn(
                        "Predicted LTV (INR)", format="₹ %.2f"
                    ),
                    "lower_80_inr": st.column_config.NumberColumn(
                        "Lower 80% range (INR)", format="₹ %.2f"
                    ),
                    "upper_80_inr": st.column_config.NumberColumn(
                        "Upper 80% range (INR)", format="₹ %.2f"
                    ),
                },
            )
            st.download_button(
                "Download predictions CSV",
                predictions.to_csv(index=False).encode("utf-8"),
                file_name="d180_ltv_predictions.csv",
                mime="text/csv",
            )
            st.caption(
                "The range is an empirical estimate from synthetic data, not a guarantee. "
                "This model is for demonstration and is not trained on CEL production data."
            )
        except (KeyError, TypeError, ValueError) as error:
            st.error(str(error))


if __name__ == "__main__":
    main()
