# CEL Analyst Assignment — End-to-End Reference

This project is a reference implementation using synthetic user data.

## Start here: Streamlit app

From the project folder, install dependencies and open the app:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`. Its first row contains a randomly sampled synthetic example. Replace those values before making a prediction.

### Input file format

For the exact column names and example values, open [`data/test_users_for_prediction.csv`](data/test_users_for_prediction.csv) in the repository. You can edit this file and upload it in the app, or make a copy and add your own rows. The app loads uploaded rows into the editable table so you can review or correct them before predicting. [`data/new_users_template.csv`](data/new_users_template.csv) is a smaller template with one example row for each product.

Each input row needs:

- `test_user_id`: a unique ID for the user.
- `product`: exactly `subscription` or `ad_supported`.
- `cac`: positive acquisition cost per user in INR. This is required for ROAS.
- Common Day 0–7 model features shown in the CSV headers.
- Product-specific features: trial and subscription fields for `subscription`, or ad fields for `ad_supported`.

`campaign` is optional. It labels users for campaign-level summaries. Leave unused product-specific fields blank. Leave `days_to_trial` blank if the subscription user did not start a trial. Hover over the app’s column headings for descriptions, or expand **Feature descriptions**. Those definitions come from `data/CEL_LTV_Feature_Dictionary.xlsx`.

The app returns per-user predicted D180 LTV and ROAS, plus empirical 80% ranges. ROAS is predicted D180 LTV divided by CAC. If campaign labels are supplied, the app also reports aggregate campaign ROAS by product and campaign. CAC and campaign are economic/context inputs; they are not used to predict LTV. You can download the results as a CSV.

The saved models are included at `models/cel_d180_ltv.joblib`. If the file is missing, train it from the project folder with `python train_model.py` before starting the app.

## Command-line predictions

The same saved models can be used without Streamlit. The input CSV must include the required columns described above.

```bash
# Predict all sample users
python predict_ltv.py data/test_users_for_prediction.csv --output outputs/test_user_predictions.csv

# Predict one user by ID
python predict_ltv.py data/test_users_for_prediction.csv --user-id TEST-004 --output outputs/one_user_prediction.csv
```

The command writes per-user LTV and ROAS to the requested output CSV. When campaign labels are present, it also creates a matching `*_campaign_summary.csv` file with aggregate predicted revenue, acquisition cost, and ROAS.

## Model and analysis

The project:

- Uses Day 0–7 behavior to predict D180 LTV separately for subscription and ad-supported products.
- Selects models using cross-validation and reports holdout metrics.
- Estimates empirical 80% prediction ranges.
- Compares campaign-level predicted LTV and ROAS.
- Exports validation metrics and feature associations under `outputs/`.

To retrain the saved models after changing the training data or model selection, run:

```bash
python train_model.py
```

If you are using the notebook, run its setup, data, and modeling cells first. Then run Streamlit from a terminal opened at the project folder.

## Important

This is not a production model and does not use CEL production data. Results are illustrative because the training data and outcome-generation process are synthetic.
