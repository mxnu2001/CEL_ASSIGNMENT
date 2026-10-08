#!/usr/bin/env python3
"""Manual, row-based Streamlit interface for the saved D180 LTV models."""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from predict_ltv import predict_dataframe, summarize_campaign_roas


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
    st.info(
        "For the required CSV headers and example rows, refer to "
        "`data/test_users_for_prediction.csv` in the repository. You can edit that file "
        "or upload your own CSV using the same format."
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
    input_columns = ["test_user_id", "product", "campaign", "cac"] + all_features

    uploaded_file = st.file_uploader(
        "Upload user inputs as a CSV (optional)", type=["csv"]
    )
    uploaded_rows = None
    editor_key = "ltv_user_inputs_example"
    if uploaded_file is not None:
        try:
            uploaded_data = pd.read_csv(uploaded_file)
            if uploaded_data.empty:
                raise ValueError("The uploaded CSV has no user rows.")
            if "test_user_id" not in uploaded_data.columns:
                if "user_id" in uploaded_data.columns:
                    uploaded_data = uploaded_data.rename(
                        columns={"user_id": "test_user_id"}
                    )
                else:
                    uploaded_data.insert(
                        0,
                        "test_user_id",
                        [f"Upload-{index:03d}" for index in range(1, len(uploaded_data) + 1)],
                    )
            if "product" not in uploaded_data.columns:
                uploaded_data["product"] = None
            if "cac" not in uploaded_data.columns and "CAC" in uploaded_data.columns:
                uploaded_data = uploaded_data.rename(columns={"CAC": "cac"})
            uploaded_rows = uploaded_data.reindex(columns=input_columns)
            file_signature = hashlib.sha256(uploaded_file.getvalue()).hexdigest()[:12]
            editor_key = f"ltv_user_inputs_upload_{file_signature}"
            st.success(
                f"Loaded {len(uploaded_rows)} row(s) from {uploaded_file.name}. "
                "Review or edit them in the input table below."
            )
            st.caption(
                "Unused CSV columns are ignored. Missing model-input columns and CAC appear "
                "blank and can be filled in the table."
            )
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
            st.error(f"Could not load this CSV: {error}")
            st.stop()

    if "ltv_placeholder_row" not in st.session_state:
        synthetic_data = pd.read_csv(ROOT / "data" / "synthetic_users.csv")
        example_product = synthetic_data["product"].sample(n=1).iloc[0]
        example = synthetic_data.loc[
            synthetic_data["product"] == example_product
        ].sample(n=1).iloc[0]
        placeholder_row = {column: None for column in input_columns}
        placeholder_row["test_user_id"] = "Example-001"
        placeholder_row["product"] = example_product
        placeholder_row["campaign"] = example.get("campaign")
        placeholder_row["cac"] = example.get("cac")
        active_features = set(artifact["products"][example_product]["features"])
        for feature in all_features:
            if (
                feature in active_features
                and feature in example.index
                and pd.notna(example[feature])
            ):
                placeholder_row[feature] = example[feature]
        st.session_state["ltv_placeholder_row"] = placeholder_row
    example_rows = pd.DataFrame(
        [st.session_state["ltv_placeholder_row"]], columns=input_columns
    )
    initial_rows = uploaded_rows if uploaded_rows is not None else example_rows

    column_config = {
        "test_user_id": st.column_config.TextColumn(
            "User ID", help="Unique label used to match each output to its input row."
        ),
        "product": st.column_config.SelectboxColumn(
            "Product",
            help="Choose the model that matches the user's product.",
            options=["subscription", "ad_supported"],
        ),
        "campaign": st.column_config.TextColumn(
            "Campaign",
            help="Optional campaign label used to calculate campaign-level ROAS summaries.",
        ),
        "cac": st.column_config.NumberColumn(
            "CAC (INR/user)",
            help="Acquisition cost per user. Required for ROAS. This is an economic input, not an LTV model feature.",
            min_value=0.01,
            step=1.0,
            format="₹ %.2f",
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
        description_rows.extend(
            [
                {
                    "Input": "Campaign",
                    "Description": "Optional acquisition campaign label used to group campaign-level ROAS results.",
                },
                {
                    "Input": "CAC (INR/user)",
                    "Description": "Positive acquisition cost per user. ROAS is predicted D180 LTV divided by CAC; CAC is not used to predict LTV.",
                },
            ]
        )
        st.dataframe(pd.DataFrame(description_rows), hide_index=True, width="stretch")
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
        "product-specific fields blank. Enter a positive CAC for every row to calculate ROAS."
    )
    entered = st.data_editor(
        initial_rows,
        num_rows="dynamic",
        column_config=column_config,
        hide_index=True,
        width="stretch",
        key=editor_key,
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
            preferred_columns = [
                "test_user_id",
                "product",
                "campaign",
                "cac_inr",
                "model",
                "predicted_d180_ltv_inr",
                "lower_80_inr",
                "upper_80_inr",
                "predicted_d180_roas",
                "lower_80_roas",
                "upper_80_roas",
            ]
            predictions = predictions[
                [column for column in preferred_columns if column in predictions.columns]
            ]
            st.subheader("Predicted D180 LTV and ROAS")
            st.dataframe(
                predictions,
                hide_index=True,
                width="stretch",
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
                    "cac_inr": st.column_config.NumberColumn(
                        "CAC (INR/user)", format="₹ %.2f"
                    ),
                    "predicted_d180_roas": st.column_config.NumberColumn(
                        "Predicted D180 ROAS (x)", format="%.2f"
                    ),
                    "lower_80_roas": st.column_config.NumberColumn(
                        "Lower 80% ROAS (x)", format="%.2f"
                    ),
                    "upper_80_roas": st.column_config.NumberColumn(
                        "Upper 80% ROAS (x)", format="%.2f"
                    ),
                },
            )
            campaign_summary = summarize_campaign_roas(predictions)
            if not campaign_summary.empty:
                st.subheader("Campaign-level ROAS")
                st.dataframe(
                    campaign_summary,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "predicted_d180_roas": st.column_config.NumberColumn(
                            "Predicted D180 ROAS (x)", format="%.2f"
                        ),
                        "lower_80_roas": st.column_config.NumberColumn(
                            "Lower 80% ROAS (x)", format="%.2f"
                        ),
                        "upper_80_roas": st.column_config.NumberColumn(
                            "Upper 80% ROAS (x)", format="%.2f"
                        ),
                        "predicted_revenue_inr": st.column_config.NumberColumn(
                            "Predicted revenue (INR)", format="₹ %.2f"
                        ),
                        "acquisition_cost_inr": st.column_config.NumberColumn(
                            "Acquisition cost (INR)", format="₹ %.2f"
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
                "ROAS is predicted D180 LTV divided by CAC. Campaign summaries aggregate "
                "predicted revenue and acquisition cost by product and campaign. The range "
                "is an empirical estimate from synthetic data, not a guarantee. "
                "This model is for demonstration and is not trained on CEL production data."
            )
        except (KeyError, TypeError, ValueError) as error:
            st.error(str(error))


if __name__ == "__main__":
    main()
