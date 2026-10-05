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
python notebooks/cel_ltv_model.py
```

## Important
This is NOT a production model and is NOT CEL's real data. The synthetic outcome-generating process is intentionally constructed for learning. You should change the assumptions, features, model and narrative before submitting.

## Core flow
D0–D7 behaviour -> predicted D180 LTV -> predicted D180 revenue -> D180 ROAS -> uncertainty -> campaign comparison.
