# CEL Analyst Assignment — D180 LTV Prediction

## 1. Executive Summary

The objective of this project is to predict **Day-180 Customer Lifetime Value (D180 LTV)** using only information available during the first 7 days of a user's lifecycle, and then use those predictions to compare the economics of two acquisition campaigns.

The analysis covers two monetization types:

- Subscription
- Ad-supported

Because no CEL production user-level dataset was provided, I created a synthetic dataset of 12,000 users to demonstrate the complete modeling workflow.

The primary dataset used for exploratory analysis, model comparison, training and evaluation is:

```text
data/synthetic_users.csv
```

Separate models were trained for subscription and ad-supported users because the two products generate value through different monetization mechanisms.

The selected models were:

- **Subscription → Ridge Regression**
- **Ad-supported → Linear Regression**

Model selection was based on 5-fold cross-validation using Mean Absolute Error (MAE), followed by evaluation on an untouched holdout set.

The selected models achieved:

| Product | MAE | RMSE | R² |
|---|---:|---:|---:|
| Subscription | 30.68 | 38.66 | 0.927 |
| Ad-supported | 18.15 | 22.62 | 0.923 |

The predicted D180 LTV is then combined with CAC to calculate predicted ROAS and compare campaign economics.

---

## 2. Problem Framing

The central question is:

> **Can early user behaviour during Days 0–7 provide enough information to estimate D180 LTV and support acquisition decisions?**

The model is therefore restricted to information that would be available within the first 7 days.

The target is:

```text
D180 LTV
```

while the inputs consist of early behavioural and monetization signals.

The overall decision workflow is:

```text
Day 0–Day 7 behaviour
        ↓
Feature engineering
        ↓
Product-specific LTV model
        ↓
Predicted D180 LTV
        ↓
Combine with CAC
        ↓
Predicted ROAS
        ↓
Campaign comparison
```

---

## 3. Data and Feature Strategy

The primary dataset used in the analysis is:

```text
data/synthetic_users.csv
```

It contains synthetic users across subscription and ad-supported products and Campaign A/B.

The features were selected based on the requirement that they should be calculable using information available by Day 7.

### Common behavioural features

The models use early behavioural signals such as:

- `active_days_7`
- `sessions_7`
- `avg_session_min_7`
- `sessions_per_active_day_7`
- `d1_retained`
- `d3_retained`
- `d7_retained`
- `days_since_last_active_7`
- `session_momentum_7`
- `median_gap_hours_7`
- `social_actions_7`

### Subscription-specific features

Subscription users additionally use:

- `trial_started_d7`
- `subscription_started_d7`
- `days_to_trial`

### Ad-supported features

Ad-supported users additionally use:

- `ad_impressions_7`
- `ad_views_7`
- `rewarded_ads_watched_7`
- `ad_view_rate_7`
- `rewarded_ad_rate_7`

Separate product-specific models were used because subscription and ad-supported users have different paths to monetization.

---

## 4. Why CAC and Campaign Are Not Predictive Features

I deliberately excluded CAC and campaign from the LTV prediction features.

### CAC

CAC is an economic input rather than a behavioural signal.

The model first answers:

```text
What is this user's expected D180 LTV?
```

CAC is then used to answer:

```text
Is acquiring this user economically attractive?
```

The user-level ROAS calculation is:

```text
ROAS = Predicted D180 LTV / CAC
```

At campaign level:

```text
Predicted Revenue = Sum of predicted D180 LTV
Acquisition Cost = Sum of CAC
Predicted ROAS = Predicted Revenue / Acquisition Cost
```

Keeping CAC outside the LTV model makes the distinction between **user value prediction** and **acquisition economics** explicit.

### Campaign

Campaign is treated as a downstream business-analysis dimension rather than a predictive feature.

The model estimates LTV from early user behaviour, and the resulting predictions are then aggregated by campaign.

This makes the workflow useful for comparing campaign economics without requiring the LTV model itself to learn campaign-specific effects.

---

## 5. Exploratory Data Analysis

The notebook contains exploratory analysis of the synthetic dataset, including:

- product-level distributions
- campaign-level comparisons
- feature distributions
- relationships between early behaviour and D180 LTV
- feature associations
- comparison of early signals across monetization types

The EDA was primarily used to understand which early behaviours were plausible predictors and to identify differences between subscription and ad-supported users before modeling.

The analysis also helped identify that different early signals matter for the two monetization mechanisms, supporting the use of separate product-specific models.

---

## 6. Modeling Strategy

I compared the following candidate models:

- Mean baseline
- Linear Regression
- Ridge Regression
- Random Forest
- HistGradientBoosting

A 5-fold cross-validation strategy was used on the training data.

The primary selection metric was **Mean Absolute Error (MAE)**.

### Why MAE?

MAE is easy to interpret because it expresses the average prediction error directly in LTV units.

For example, an MAE of 30 means that the model's predictions are off by approximately 30 LTV units on average.

MAE was preferred as the primary selection metric because it is less sensitive to extreme errors than RMSE while remaining directly interpretable for the business problem.

---

## 7. Model Selection Results

The selected models were:

| Product | Selected Model | CV MAE |
|---|---|---:|
| Subscription | Ridge Regression | 31.83 |
| Ad-supported | Linear Regression | 18.56 |

For subscription, Ridge Regression produced the best cross-validation MAE by a very small margin over Linear Regression.

For ad-supported users, Linear Regression produced the lowest cross-validation MAE, again with Ridge performing almost identically.

The more complex tree-based models did not provide enough improvement to justify their additional complexity on this synthetic dataset.

This led to the choice of simpler, interpretable models.

---

## 8. Holdout Validation

After model selection, the selected models were evaluated on an untouched holdout set.

| Product | Model | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Subscription | Ridge Regression | 30.68 | 38.66 | 0.927 |
| Ad-supported | Linear Regression | 18.15 | 22.62 | 0.923 |

Both models achieved an R² above 0.92 on the synthetic holdout data.

However, these metrics should be interpreted cautiously because the dataset is synthetic and the outcome-generation process may not represent real customer behaviour.

---

## 9. Handling Missing Values

The prediction pipeline validates the required input columns before generating predictions.

For `days_to_trial`, blank values are allowed because not every user necessarily starts a trial during the first 7 days.

The prediction workflow therefore distinguishes between a valid non-conversion signal and an invalid or missing required input rather than treating every missing value as an error.

This is particularly important when applying the model to new users, because some product-specific behaviours naturally do not occur for every user.

---

## 10. Prediction Uncertainty

A point prediction alone may not be sufficient for acquisition-budget decisions.

The reusable prediction pipeline therefore produces an empirical **80% prediction range** around the predicted D180 LTV.

The uncertainty range is derived from residuals generated through 5-fold out-of-fold predictions.

For each user, the system can provide:

- predicted D180 LTV
- lower 80% LTV estimate
- upper 80% LTV estimate
- predicted ROAS
- lower 80% ROAS estimate
- upper 80% ROAS estimate

These ranges should be interpreted as empirical uncertainty estimates rather than formally calibrated probabilistic prediction intervals.

---

## 11. Campaign-Level LTV and ROAS

The campaign-level analysis aggregates user predictions.

For each campaign:

```text
Predicted Revenue
= Sum of predicted D180 LTV

Acquisition Cost
= Sum of CAC

Predicted ROAS
= Predicted Revenue / Acquisition Cost
```

This provides a direct connection between the LTV model and the business decision of where acquisition budget should be allocated.

---

## 12. Campaign Results on the Synthetic Holdout

### Subscription

| Campaign | Users | Predicted D180 LTV | Predicted ROAS |
|---|---:|---:|---:|
| Campaign A | 740 | 278.08 | 3.09 |
| Campaign B | 769 | 296.98 | 2.83 |

Campaign B produces higher predicted D180 LTV, but Campaign A produces higher predicted ROAS because Campaign B has a higher CAC.

### Ad-supported

| Campaign | Users | Predicted D180 LTV | Predicted ROAS |
|---|---:|---:|---:|
| Campaign A | 731 | 180.41 | 2.00 |
| Campaign B | 761 | 189.61 | 1.81 |

The same pattern occurs for the ad-supported product.

Campaign B produces higher predicted LTV but lower predicted ROAS because the increase in CAC is greater than the corresponding improvement in predicted LTV.

This demonstrates why campaign decisions should consider **both long-term value and acquisition cost**.

---

## 13. D7 vs D180 Campaign Reversal

One of the key questions in the assignment is whether the campaign that appears stronger using early signals necessarily remains stronger when evaluated on longer-term value.

The analysis shows why D7 performance and D180 economics should not be treated as identical measures.

Early behaviour is useful because it provides predictive information about future value.

However, a campaign can generate stronger early engagement or higher predicted LTV while still being economically inferior if its acquisition cost is sufficiently higher.

Therefore, the final campaign decision should consider:

- predicted D180 LTV
- CAC
- predicted ROAS
- uncertainty around the predictions

rather than relying only on D7 engagement metrics.

---

## 14. Reusable Prediction Workflow

The project was designed so that the trained models can be reused with new user-level data.

The workflow is:

```text
New Day-0–Day-7 user data
          ↓
Input validation
          ↓
Product-specific model
          ↓
Predicted D180 LTV
          ↓
Empirical 80% uncertainty range
          ↓
CAC
          ↓
Predicted ROAS
          ↓
Campaign aggregation
```

The project provides both a command-line prediction script and a Streamlit interface.

The Python prediction workflow can be run using:

```bash
python predict_ltv.py data/new_users_template.csv
```

The interactive interface can be launched using:

```bash
streamlit run streamlit_app.py
```

This allows new users to be scored without modifying the underlying modeling code.

---

## 15. Key Assumptions

The main assumptions are:

1. All predictive features are available by Day 7.
2. D180 LTV is treated as the future outcome to be predicted.
3. Subscription and ad-supported users require separate models because their monetization mechanisms differ.
4. CAC is used for ROAS calculation rather than LTV prediction.
5. Campaign is used for downstream campaign analysis rather than as an LTV feature.
6. The empirical uncertainty ranges are intended as practical uncertainty estimates, not calibrated probability intervals.
7. The synthetic dataset is only a demonstration of the modeling workflow and does not represent CEL production behaviour.

---

## 16. Validation With Real Data

If real CEL production data became available, I would first validate the data-generating process and ensure that every feature is genuinely available by Day 7.

I would then:

1. Define historical prediction cohorts.
2. Prevent post-Day-7 leakage.
3. Use a time-based validation strategy.
4. Evaluate MAE, RMSE and R².
5. Evaluate performance separately by product.
6. Evaluate performance across campaigns and acquisition cohorts.
7. Test prediction calibration and uncertainty coverage.
8. Compare predicted versus realized D180 LTV after cohorts mature.
9. Monitor model performance over time.
10. Investigate data drift and changes in monetization behaviour.

A time-based split would be particularly important in production because acquisition channels, pricing, user behaviour and monetization can change over time.

---

## 17. Potential Improvements

With sufficient real data, the model could be improved through:

- richer behavioural features
- cohort and acquisition-source information
- more granular retention features
- better treatment of skewed LTV distributions
- gradient-boosting models if they provide measurable improvement
- calibrated prediction intervals
- time-to-event or survival-based approaches
- model monitoring and drift detection

However, I would only introduce additional complexity if it improved validation performance or decision quality.

The current approach intentionally favours a simpler model where it performs adequately.

---

## 18. Limitations

The most important limitation is that the entire analysis uses synthetic data because no CEL production user-level data was supplied.

Therefore:

- model performance is illustrative
- campaign results are illustrative
- the synthetic outcome-generation process may not represent real users
- uncertainty estimates are empirical rather than formally calibrated
- campaign comparisons should not be interpreted as causal estimates
- production deployment would require validation using real historical cohorts

The current project should therefore be viewed as a demonstration of the methodology and reusable workflow rather than a production-ready LTV model.

---

## 19. AI Workflow and Disclosure

AI tools were used during development as coding and reasoning assistants.

The project folder was treated as the source of truth. I remained responsible for analytical decisions including:

- feature selection
- leakage checks
- model comparison
- validation methodology
- interpretation of results
- final project structure

The generated code and analytical outputs were reviewed before being included in the submission.

---

## 20. Final Takeaway

The project demonstrates an end-to-end approach for estimating D180 LTV from early user behaviour and translating those predictions into campaign-level economics.

The main principles are:

1. Use only information available by Day 7.
2. Model subscription and ad-supported products separately.
3. Keep CAC separate from the LTV prediction model.
4. Use cross-validation to select the model rather than assuming a complex model is better.
5. Evaluate the final models on an untouched holdout set.
6. Include uncertainty around predictions.
7. Translate predicted LTV into ROAS for campaign comparison.

The synthetic results demonstrate that early user behaviour can provide useful signals for estimating longer-term value.

However, the final acquisition decision should consider **predicted LTV, CAC, ROAS and uncertainty together**, rather than relying on early engagement or predicted LTV alone.

The next step for a production implementation would be to validate the methodology on real CEL historical cohorts and monitor prediction accuracy and campaign economics over time.