# %% [markdown]
# # Customer Lifetime Value Model
#
# This notebook-style script generates synthetic Day 0–7 user data, trains a
# separate D180 LTV model for each product, and compares campaign performance.
# Run the file from any working directory with `python notebooks/cel_ltv_model.py`.

# %% [markdown]
# ## 1. Imports and project setup

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(42)

# Assumed campaign-level acquisition costs; CAC is an economic input, not a
# user-level feature. Campaign names are placeholders: the assignment does not
# define what Campaign A and Campaign B represent in the real world.
CAMPAIGN_CAC = {"Campaign_A": 90, "Campaign_B": 105}

# %% [markdown]
# ## 2. Generate synthetic user data
#
# The data includes acquisition cost, early engagement, product-specific
# conversion or ad activity, and a synthetic D180 lifetime value outcome.

# %%
def sigmoid(x):
    """Convert scores to probabilities."""
    return 1 / (1 + np.exp(-x))


def make_data(n=12_000):
    """Create a reproducible synthetic cohort of user-level observations."""
    product = rng.choice(["subscription", "ad_supported"], n)
    campaign = rng.choice(["Campaign_A", "Campaign_B"], n)
    # These campaign groups represent assumed acquisition strategies. The
    # synthetic strategy effects below are illustrative, not CEL facts.
    quality = rng.normal(0, 1, n)

    active_days = np.clip(
        np.rint(
            3.3
            + 0.9 * quality
            + np.where(campaign == "Campaign_A", -0.25, 0.25)
            + rng.normal(0, 1, n)
        ),
        1,
        7,
    ).astype(int)
    sessions = np.clip(
        np.rint(
            5
            + 2.2 * quality
            + 0.9 * active_days
            + np.where(campaign == "Campaign_A", 0.8, 0)
            + rng.normal(0, 2.5, n)
        ),
        1,
        35,
    ).astype(int)
    avg_session_min = np.clip(
        7 + 2.2 * quality + 0.5 * active_days + rng.normal(0, 2.5, n),
        2,
        25,
    )
    engagement = (
        0.45 * active_days / 7
        + 0.35 * np.minimum(sessions / 20, 1)
        + 0.20 * np.minimum(avg_session_min / 20, 1)
    )

    trial_started = (
        (product == "subscription")
        & (
            rng.random(n)
            < sigmoid(
                -1
                + 2.2 * engagement
                + 0.75 * quality
                + np.where(campaign == "Campaign_A", -0.25, 0.25)
            )
        )
    ).astype(int)
    subscription_started = (
        (product == "subscription")
        & (
            rng.random(n)
            < sigmoid(-2 + 2.6 * engagement + 1.1 * trial_started + 0.65 * quality)
        )
    ).astype(int)

    ad_impressions = np.where(
        product == "ad_supported",
        np.clip(
            np.rint(5 + 3.2 * sessions + 2.5 * quality + rng.normal(0, 7, n)),
            0,
            180,
        ),
        0,
    ).astype(int)
    ad_views = np.where(
        product == "ad_supported",
        np.clip(np.rint(0.70 * ad_impressions + rng.normal(0, 4, n)), 0, None),
        0,
    ).astype(int)

    subscription_ltv = (
        25
        + 145 * subscription_started
        + 38 * trial_started
        + 28 * active_days
        + 3 * sessions
        + 8 * avg_session_min
        + 55 * quality
    )
    ad_supported_ltv = (
        12
        + 0.95 * ad_impressions
        + 0.45 * ad_views
        + 18 * active_days
        + 2.8 * sessions
        + 5 * avg_session_min
        + 28 * quality
    )
    ltv = np.where(product == "subscription", subscription_ltv, ad_supported_ltv)
    ltv += np.where(campaign == "Campaign_A", 12, 35)
    ltv = np.clip(
        ltv + rng.normal(0, np.where(product == "subscription", 35, 18), n),
        0,
        None,
    )

    return pd.DataFrame(
        {
            "user_id": np.arange(1, n + 1),
            "product": product,
            "campaign": campaign,
            "active_days_7": active_days,
            "sessions_7": sessions,
            "avg_session_min_7": avg_session_min,
            "d7_engagement_score": engagement,
            "trial_started_d7": trial_started,
            "subscription_started_d7": subscription_started,
            "ad_impressions_7": ad_impressions,
            "ad_views_7": ad_views,
            "d180_ltv": ltv,
        }
    )


df = make_data()
df.to_csv(DATA_DIR / "synthetic_users.csv", index=False)
df.head()

# %% [markdown]
# ## 3. Select product-specific model features

# %%
def features(data, product):
    """Return the early-life features used for the selected product."""
    common_features = [
        "active_days_7",
        "sessions_7",
        "avg_session_min_7",
        "d7_engagement_score",
    ]
    product_features = (
        ["trial_started_d7", "subscription_started_d7"]
        if product == "subscription"
        else ["ad_impressions_7", "ad_views_7"]
    )
    return data[common_features + product_features]

# %% [markdown]
# ## 4. Train models and calculate validation metrics
#
# Each product gets a mean-value baseline and a histogram gradient boosting
# model. The empirical 10th and 90th percentiles of validation residuals form
# the prediction range used below.

# %%
summary = []
metrics = []
associations = []

for product in ["subscription", "ad_supported"]:
    product_data = df[df["product"] == product].copy()
    X = features(product_data, product)
    y = product_data["d180_ltv"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    model = HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=2,
        random_state=42,
    ).fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics.extend(
        [
            {
                "product": product,
                "model": "mean_baseline",
                "MAE": mean_absolute_error(y_test, baseline_predictions),
                "RMSE": mean_squared_error(y_test, baseline_predictions) ** 0.5,
                "R2": r2_score(y_test, baseline_predictions),
            },
            {
                "product": product,
                "model": "hist_gradient_boosting",
                "MAE": mean_absolute_error(y_test, predictions),
                "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
                "R2": r2_score(y_test, predictions),
            },
        ]
    )

    residual_quantiles = np.quantile(y_test.to_numpy() - predictions, [0.10, 0.90])
    product_data["predicted_d180_ltv"] = np.clip(model.predict(X), 0, None)
    product_data["lower_80"] = np.clip(
        product_data["predicted_d180_ltv"] + residual_quantiles[0], 0, None
    )
    product_data["upper_80"] = np.clip(
        product_data["predicted_d180_ltv"] + residual_quantiles[1], 0, None
    )

    for campaign in ["Campaign_A", "Campaign_B"]:
        campaign_data = product_data[product_data["campaign"] == campaign]

        if product == "subscription":
            value_proxy = (
                20 * campaign_data["trial_started_d7"].mean()
                + 50 * campaign_data["subscription_started_d7"].mean()
            )
        else:
            value_proxy = (
                0.20 * campaign_data["ad_impressions_7"].mean()
                + 0.35 * campaign_data["ad_views_7"].mean()
            )

        predicted_ltv = campaign_data["predicted_d180_ltv"].mean()
        predicted_revenue = campaign_data["predicted_d180_ltv"].sum()
        actual_ltv = campaign_data["d180_ltv"].mean()
        actual_revenue = campaign_data["d180_ltv"].sum()
        campaign_cac = CAMPAIGN_CAC[campaign]
        acquisition_spend = len(campaign_data) * campaign_cac
        predicted_roas = predicted_revenue / acquisition_spend

        summary.append(
            {
                "product": product,
                "campaign": campaign,
                "users": len(campaign_data),
                "campaign_cac": campaign_cac,
                "acquisition_spend": acquisition_spend,
                "d7_value_proxy": value_proxy,
                "predicted_d180_ltv": predicted_ltv,
                "predicted_ltv_lower_80": campaign_data["lower_80"].mean(),
                "predicted_ltv_upper_80": campaign_data["upper_80"].mean(),
                "predicted_d180_revenue": predicted_revenue,
                "predicted_d180_roas": predicted_roas,
                "predicted_d180_roas_lower_80": campaign_data["lower_80"].sum()
                / acquisition_spend,
                "predicted_d180_roas_upper_80": campaign_data["upper_80"].sum()
                / acquisition_spend,
                "actual_d180_ltv_synthetic": actual_ltv,
                "actual_d180_revenue_synthetic": actual_revenue,
                "actual_d180_roas_synthetic": actual_revenue / acquisition_spend,
            }
        )

    for column in X.columns:
        associations.append(
            {
                "product": product,
                "feature": column,
                "spearman_corr_with_d180_ltv": X[column].corr(y, method="spearman"),
            }
        )

# %% [markdown]
# ## 5. Export results

# %%
summary_df = pd.DataFrame(summary)
metrics_df = pd.DataFrame(metrics)
associations_df = pd.DataFrame(associations)

summary_df.to_csv(OUTPUT_DIR / "campaign_summary.csv", index=False)
metrics_df.to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)
associations_df.sort_values(
    ["product", "spearman_corr_with_d180_ltv"], ascending=[True, False]
).to_csv(OUTPUT_DIR / "feature_associations.csv", index=False)

print(summary_df.round(2).to_string(index=False))
print("\nValidation\n", metrics_df.round(3).to_string(index=False))
