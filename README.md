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

## Project Structure

```text
CEL_ASSIGNMENT/
│
├── data/
│   ├── synthetic_users.csv
│   ├── new_users_template.csv
│   ├── test_users_for_prediction.csv
│   └── CEL_LTV_Feature_Dictionary.xlsx
│
├── models/
│   └── cel_d180_ltv.joblib
│
├── notebooks/
│   └── CEL_Customer_Lifetime_Value_Model_No_Game_Metrics_EDA_Clean.ipynb
│
├── outputs/
│   ├── model_selection_cv_metrics.csv
│   ├── selected_model_holdout_metrics.csv
│   ├── campaign_summary.csv
│   └── feature_associations.csv
│
├── train_model.py
├── predict_ltv.py
├── streamlit_app.py
├── requirements.txt
├── README.md
└── writeup/
    └── CEL_Analyst_Assignment_Writeup.md
```

### File and Data Description

| File / Folder | Purpose |
|---|---|
| `data/synthetic_users.csv` | **Primary dataset used for EDA, model comparison, training and evaluation** |
| `data/new_users_template.csv` | Template for entering new users and generating predictions |
| `data/test_users_for_prediction.csv` | Example input for testing the prediction pipeline |
| `data/CEL_LTV_Feature_Dictionary.xlsx` | Definitions and descriptions of the model features |
| `models/cel_d180_ltv.joblib` | Saved trained model artifact used by the prediction pipeline |
| `notebooks/` | Complete exploratory analysis, feature analysis, model comparison and validation |
| `outputs/` | Saved model-selection, holdout-validation and campaign-analysis results |
| `train_model.py` | Re-trains the selected models and saves the model artifact |
| `predict_ltv.py` | Generates D180 LTV and ROAS predictions for new users |
| `streamlit_app.py` | Interactive interface for entering/uploading users and viewing predictions |
| `writeup/` | Short analytical explanation of the approach, assumptions and findings |

### Primary Dataset

The main dataset used throughout the project is:

```text
data/synthetic_users.csv
```

This is the dataset used for:

- Exploratory Data Analysis
- Feature analysis
- Model comparison
- Cross-validation
- Model training
- Holdout evaluation
- Campaign-level analysis

The dataset is synthetic because no CEL production user-level dataset was provided with the assignment. Therefore, the model performance and campaign results should be interpreted as illustrative rather than production estimates.

The other CSV files have different purposes:

- `new_users_template.csv` → intended for entering new users for prediction
- `test_users_for_prediction.csv` → example/test input for demonstrating the prediction pipeline
- `synthetic_users.csv` → the actual dataset used to develop and evaluate the models

### Model Architecture

The project trains separate models for the two monetization types:

```text
                    Day 0–Day 7 Data
                           │
              ┌────────────┴────────────┐
              │                         │
        Subscription              Ad-supported
              │                         │
        Ridge Regression        Linear Regression
              │                         │
              └────────────┬────────────┘
                           │
                   Predicted D180 LTV
                           │
                          CAC
                           │
                   Predicted ROAS
                           │
                  Campaign Comparison
```

The separation is intentional because subscription and ad-supported products generate value through different monetization mechanisms.

### Why CAC Is Not an LTV Feature

CAC is treated as an economic input rather than an LTV prediction feature.

The model first estimates:

```text
Predicted D180 LTV
```

and CAC is then used to calculate:

```text
ROAS = Predicted D180 LTV / CAC
```

At campaign level:

```text
Predicted Revenue = Sum of predicted D180 LTV
Acquisition Cost = Sum of CAC
Predicted ROAS = Predicted Revenue / Acquisition Cost
```

This keeps the user-level LTV prediction separate from acquisition economics.

### Why Campaign Is Not a Predictive Feature

Campaign is used for downstream campaign-level analysis rather than as a direct LTV model feature.

This allows the model to estimate user value from early behaviour and then compare the resulting economics across campaigns.

### Selected Models

Model selection was performed using 5-fold cross-validation with MAE as the primary metric.

The selected models are:

| Product | Selected Model |
|---|---|
| Subscription | Ridge Regression |
| Ad-supported | Linear Regression |

The selected models were preferred because they performed competitively with or better than the more complex alternatives on the synthetic dataset while remaining simple and interpretable.

### Holdout Performance

| Product | Model | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Subscription | Ridge Regression | 30.68 | 38.66 | 0.927 |
| Ad-supported | Linear Regression | 18.15 | 22.62 | 0.923 |

These metrics are based on the synthetic holdout data and should not be interpreted as production performance.

### Prediction Uncertainty

The prediction pipeline provides an empirical 80% range around the predicted D180 LTV.

The same range is translated into an ROAS range using CAC.

The uncertainty estimates are derived from residuals obtained through 5-fold out-of-fold predictions. They are intended to communicate practical prediction uncertainty and are not formally calibrated probabilistic prediction intervals.

### Example Campaign Results

On the synthetic holdout data:

| Product | Campaign | Predicted D180 LTV | Predicted ROAS |
|---|---|---:|---:|
| Subscription | A | 278.08 | 3.09 |
| Subscription | B | 296.98 | 2.83 |
| Ad-supported | A | 180.41 | 2.00 |
| Ad-supported | B | 189.61 | 1.81 |

An important observation is that Campaign B has higher predicted LTV for both products, but its higher CAC results in lower predicted ROAS.

This demonstrates why campaign decisions should consider both long-term value and acquisition cost rather than relying on predicted LTV alone.

### Important Limitation

This project is a methodology demonstration using synthetic data.

It does **not** use CEL production data, and therefore:

- model performance is illustrative
- campaign comparisons are illustrative
- the synthetic data-generation process may not represent real user behaviour
- production deployment would require validation on real historical cohorts
- prediction performance and uncertainty would need to be monitored after deployment