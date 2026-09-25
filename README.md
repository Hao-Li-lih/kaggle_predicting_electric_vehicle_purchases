# Predicting Electric Vehicle Purchases

Machine-learning experiments for the Kaggle tabular competition that predicts the probability of `Will_Buy_EV`.

The project compares classical baselines, feature-engineered gradient boosting, CatBoost, LightGBM, XGBoost, a PyTorch categorical-embedding network, spline logistic GAMs, external-data priors, and high-resolution histogram XGBoost models.

## Current result

- Best observed public leaderboard score before the fine-bin experiments: **0.94187**
- Best local holdout ROC-AUC: **0.943136**
- Best local model: XGBoost with `max_bin=8192`

Increasing the histogram resolution produced a larger and more consistent gain than adding further manual interactions. The dominant behavioral variables define strong conditional segments, while income, commute distance, and charging availability determine fine-grained ordering inside those segments.

## Repository structure

```text
notebooks/
  01_data_exploration.ipynb
  02_feature_engineering_model_comparison.ipynb
  03_advanced_models_and_deep_learning.ipynb
  04_leaderboard_optimization.ipynb
  05_model_rethinking_and_finebin_xgboost.ipynb

generate_advanced_submissions.py
optimize_xgboost_leaderboard.py
structured_model_probe.py
original_prior_probe.py
ranking_model_probe.py
train_finebin_xgboost.py
artifacts/
```

Competition data, external datasets, generated submissions, virtual environments, and model logs are intentionally excluded from Git.

## Data setup

Place the Kaggle files in `data/`:

```text
data/
  train.csv
  test.csv
  sample_submission.csv
```

The expected target column is `Will_Buy_EV`, and the identifier column is `id`.

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Main workflows

Generate the advanced baseline submissions:

```powershell
python generate_advanced_submissions.py
```

Run the first leaderboard-focused XGBoost optimization:

```powershell
python optimize_xgboost_leaderboard.py
```

Train the high-resolution XGBoost models and generate the latest candidate submissions:

```powershell
python train_finebin_xgboost.py
```

Generated CSV files are written to `submissions/` and are not committed.

## Latest candidate models

- XGBoost `max_bin=4096`, two-seed ensemble
- XGBoost `max_bin=8192`, single seed
- Probability blend of the two resolutions
- Percentile-rank blend designed for ROC-AUC

See `notebooks/05_model_rethinking_and_finebin_xgboost.ipynb` for the model-selection evidence and validation plots.
