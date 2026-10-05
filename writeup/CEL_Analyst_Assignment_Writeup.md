# Chaos Engine Labs — Analyst Assignment
## Reference Write-up

### 1. Objective
The objective is to predict D180 LTV using only information available during the first seven days after install, separately for a subscription product and an ad-supported product. The prediction is then used to estimate D180 ROAS for two acquisition campaigns.

### 2. Observation window
I use Day 0–7 as the observation window. This is the maximum window permitted by the brief and gives the model a fuller view of early engagement while remaining early enough for an acquisition decision.

### 3. Unit of analysis
One row represents one acquired user. Model features describe behaviour available by Day 7. The target is cumulative revenue per acquired user through Day 180 and is used only for model fitting and evaluation, never as an input feature. Campaign and CAC are kept out of the feature set.

### 4. Subscription product
Candidate early signals include active days, sessions, session duration, trial initiation and subscription initiation. Monetisation actions are expected to carry stronger information about long-term value than raw activity alone.

### 5. Ad-supported product
Candidate early signals include active days, sessions, session duration, ad impressions and ad views. The key difference is that future value is driven by repeated usage and opportunities to show monetised advertising.

### 6. Modeling approach
I use a simple mean predictor as a baseline and HistGradientBoostingRegressor as the main reference model. The goal is not maximum algorithmic complexity; it is a reusable and explainable early-LTV system.

### 7. Uncertainty and signal analysis
Prediction uncertainty is estimated from the 10th and 90th percentiles of residuals on a held-out validation set. The campaign output reports average predicted D180 LTV and an empirical 80% range, plus a corresponding ROAS range. Spearman associations between each eligible Day 7 feature and D180 LTV provide a simple signal analysis.

### 8. Campaign assumptions and economics
Campaign A and Campaign B are two assumed acquisition strategies; the assignment does not define what they represent in the real world. The synthetic generator gives these groups different illustrative acquisition and user-mix assumptions. Campaign-level CAC is fixed at ₹90 per acquired user for Campaign A and ₹105 for Campaign B. CAC is an economic input, not a user-level behavior or an LTV model feature.

For each product and campaign, predicted campaign revenue is the sum of users' predicted D180 LTV. Acquisition spend is the number of acquired users multiplied by that campaign's CAC. Predicted D180 ROAS is predicted campaign revenue divided by acquisition spend. Actual synthetic LTV is used only for evaluation, including the corresponding actual synthetic ROAS.

### 9. D7 vs D180
A campaign can look stronger during the first seven days yet produce lower D180 value if its early activity is high but its users have weaker long-term retention or monetisation. Conversely, a campaign with lower early volume can attract users who convert or remain active for longer.

### 10. Validation plan
When real D180 outcomes arrive, compare predicted and observed LTV by user, campaign and product. Track MAE/RMSE, calibration and systematic bias. Recalibrate or retrain when the relationship between early signals and long-term value changes.

### 11. Limitations
Synthetic data cannot establish real-world relationships. The uncertainty estimate is empirical rather than a fully probabilistic model. Campaign selection can also create confounding: acquisition source may affect user mix, pricing, creative, geography and other factors simultaneously.

### 12. AI workflow
AI can be used for terminology research, feature brainstorming, code scaffolding, debugging, test-case generation and critique. The analyst remains responsible for assumptions, leakage checks, interpretation, validation and the final submission.
