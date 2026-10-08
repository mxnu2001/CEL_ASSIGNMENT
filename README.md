# CEL Analyst Assignment — End-to-End Reference

This project is a reference implementation of the supplied assignment.

## What it does
- Generates synthetic user-level Day 0–7 data.
- Separates subscription and ad-supported products.
- Trains a simple baseline and a nonlinear regression model.
- Predicts D180 LTV from D0–D7 inputs.
- Estimates an empirical 80% prediction range.
- Compares Campaign A and Campaign B.
- Calculates campaign-level predicted D180 ROAS using fixed campaign CAC assumptions.
- Exports validation metrics and feature associations.

Campaign A and Campaign B are assumed acquisition strategies because the assignment does not define their real-world meaning. Synthetic campaign-level CAC is ₹90 per acquired user for A and ₹105 for B. CAC, campaign labels, and actual D180 LTV are excluded from the predictive feature set; actual D180 LTV is retained for evaluation.

## Run
```bash
pip install -r requirements.txt
python train_model.py
```

## Predict for new users

Training saves the selected subscription and ad-supported models to `models/cel_d180_ltv.joblib`. The models are refit on all labeled synthetic users, and the artifact includes the feature schema and empirical 80% prediction-range offsets.

Copy `data/new_users_template.csv`, replace the example Day 0–7 values with one row per user, and keep the `product` value as either `subscription` or `ad_supported`. Leave `days_to_trial` blank when a subscription user did not start a trial. Subscription rows use the trial/subscription columns; ad-supported rows use the ad columns. The unused product-specific columns may remain blank.

```bash
python predict_ltv.py my_new_users.csv --output predictions.csv
```

For a ready-to-run example, use `data/test_users_for_prediction.csv`; its example results are in `outputs/test_user_predictions.csv`.

The output includes predicted D180 LTV and an empirical 80% range in INR for each row. The prediction range describes residual variation in this synthetic validation setup; it is not a guarantee. Retrain after changing the training data or selected models.

## Important
This is NOT a production model and is NOT CEL's real data. The synthetic outcome-generating process is intentionally constructed for learning. You should change the assumptions, features, model and narrative before submitting.

## Core flow
D0–D7 behaviour -> predicted D180 LTV -> predicted D180 revenue -> D180 ROAS -> uncertainty -> campaign comparison.
