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

Run these commands from the project folder. If you are using the notebook, first run its setup/data/modeling cells, then use a notebook terminal or a cell prefixed with `!` for the commands below. The training and prediction scripts resolve their input data and saved model relative to the project folder.

```bash
# Install dependencies once, then build the saved model artifact
pip install -r requirements.txt
python train_model.py

# Run all sample users together
python predict_ltv.py data/test_users_for_prediction.csv --output outputs/test_user_predictions.csv

# Run just one sample user by ID
python predict_ltv.py data/test_users_for_prediction.csv --user-id TEST-004 --output outputs/one_user_prediction.csv
```

To predict for new users, copy `data/new_users_template.csv`, replace the example Day 0–7 values, and keep the `product` value as either `subscription` or `ad_supported`. Include a unique `test_user_id` (or `user_id`) column if you want to select one row with `--user-id`. Leave `days_to_trial` blank when a subscription user did not start a trial. Subscription rows use the trial/subscription columns; ad-supported rows use the ad columns. The unused product-specific columns may remain blank. Then run `python predict_ltv.py my_new_users.csv --output predictions.csv` to score the whole file, or add `--user-id YOUR_ID` to score one row.

The one-user option requires the selected ID to match exactly one input row. For a ready-to-run example, use `data/test_users_for_prediction.csv`; its batch results are in `outputs/test_user_predictions.csv`.

The output includes predicted D180 LTV and an empirical 80% range in INR for each row. The prediction range describes residual variation in this synthetic validation setup; it is not a guarantee. Retrain after changing the training data or selected models.

## Streamlit input page

To open a page for entering users manually, install the requirements and start the app from the project folder:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Streamlit opens the app at `http://localhost:8501`. Edit the table one user per row, choose the product, and enter the Day 0–7 model inputs. Add rows with the table control to score multiple users together. Hover over a column heading or open **Feature descriptions** for the short definitions from `data/CEL_LTV_Feature_Dictionary.xlsx`. Leave the feature columns for the other product blank. Click **Predict D180 LTV** to view the predicted value and empirical 80% range in INR, then download the results as a CSV.

The saved models must exist at `models/cel_d180_ltv.joblib`. If they do not, run `python train_model.py` before starting the app. From a notebook, run the command in a terminal opened at the project folder; the notebook itself can still be used for the training and analysis workflow.

## Important
This is NOT a production model and is NOT CEL's real data. The synthetic outcome-generating process is intentionally constructed for learning. You should change the assumptions, features, model and narrative before submitting.

## Core flow
D0–D7 behaviour -> predicted D180 LTV -> predicted D180 revenue -> D180 ROAS -> uncertainty -> campaign comparison.
