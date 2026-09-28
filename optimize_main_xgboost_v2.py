import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "Will_Buy_EV"
ID_COLUMN = "id"
RANDOM_STATE = 42


def engineer_features(data):
    frame = data.copy().drop(columns=[ID_COLUMN], errors="ignore")
    home = frame["Charging_Stations_Near_Home"]
    work = frame["Charging_Stations_Near_Work"]
    income = frame["Annual_Income_USD"]
    commute = frame["Daily_Commute_km"]
    concern = frame["Environmental_Concern_Level"]
    cars = frame["Number_of_Cars_Owned"].clip(lower=1)
    home_flag = frame["Home_Charging_Possible"].map({"No": 0.0, "Yes": 1.0})
    subsidy_flag = frame["Subsidy_Available"].map({"No": 0.0, "Yes": 1.0})
    anxiety = frame["Range_Anxiety_Level"].map(
        {"Low": 0.0, "Medium": 1.0, "High": 2.0}
    )

    frame["Total_Charging_Stations"] = home + work
    frame["Min_Charging_Stations"] = np.minimum(home, work)
    frame["Max_Charging_Stations"] = np.maximum(home, work)
    frame["Home_Work_Station_Gap"] = home - work
    frame["Commute_Per_Station"] = commute / (frame["Total_Charging_Stations"] + 1.0)
    frame["Log_Annual_Income"] = np.log1p(income)
    frame["Income_Per_Car"] = income / cars
    frame["Income_Per_Commute"] = income / commute.clip(lower=1)
    frame["Environment_Income_Interaction"] = concern * frame["Log_Annual_Income"]
    frame["Environment_Charging_Interaction"] = concern * frame["Total_Charging_Stations"]
    frame["Home_Charging_Flag"] = home_flag
    frame["Subsidy_Flag"] = subsidy_flag
    frame["Range_Anxiety_Ordinal"] = anxiety
    frame["Age_Squared"] = frame["Age"] ** 2
    frame["Income_Squared_Scaled"] = (income / 100_000.0) ** 2
    frame["Log_Daily_Commute"] = np.log1p(commute)
    frame["Charging_Balance_Ratio"] = (home + 1.0) / (work + 1.0)
    frame["Income_x_Subsidy"] = frame["Log_Annual_Income"] * subsidy_flag
    frame["Income_x_Home_Charging"] = frame["Log_Annual_Income"] * home_flag
    frame["Concern_x_Subsidy"] = concern * subsidy_flag
    frame["Concern_x_Home_Charging"] = concern * home_flag
    frame["Anxiety_x_Commute"] = anxiety * commute
    frame["Anxiety_x_Charging"] = anxiety * frame["Total_Charging_Stations"]
    frame["Concern_Minus_Anxiety"] = concern - anxiety
    frame["Core_Context"] = (
        concern.astype(str)
        + "_"
        + frame["Subsidy_Available"].astype(str)
        + "_"
        + frame["Range_Anxiety_Level"].astype(str)
        + "_"
        + frame["Home_Charging_Possible"].astype(str)
    )
    return frame.replace([np.inf, -np.inf], np.nan)


def metrics(target, probabilities):
    return {
        "ROC_AUC": roc_auc_score(target, probabilities),
        "Log_Loss": log_loss(target, probabilities),
        "Average_Precision": average_precision_score(target, probabilities),
    }


def percentile_rank(values):
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


ARTIFACT_DIR.mkdir(exist_ok=True)
print("Loading and splitting data", flush=True)
data = pd.read_csv(DATA_DIR / "train.csv")
features = engineer_features(data.drop(columns=[TARGET]))
target = data[TARGET].eq("Yes").astype(int)

train_tune_features, holdout_features, train_tune_target, holdout_target = train_test_split(
    features,
    target,
    test_size=0.15,
    random_state=RANDOM_STATE,
    stratify=target,
)
train_features, tuning_features, train_target, tuning_target = train_test_split(
    train_tune_features,
    train_tune_target,
    test_size=0.17647058823529413,
    random_state=RANDOM_STATE + 1,
    stratify=train_tune_target,
)

preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), selector(dtype_include=np.number)),
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]),
        selector(dtype_exclude=np.number),
    ),
])
print("Encoding matrices", flush=True)
train_matrix = preprocessor.fit_transform(train_features)
tuning_matrix = preprocessor.transform(tuning_features)
holdout_matrix = preprocessor.transform(holdout_features)

common = {
    "max_bin": 4096,
    "tree_method": "hist",
    "eval_metric": "auc",
    "early_stopping_rounds": 150,
    "n_jobs": 4,
    "random_state": RANDOM_STATE,
}
candidate_parameters = {
    "Depth5_Baseline": {
        "n_estimators": 2200,
        "learning_rate": 0.025,
        "max_depth": 5,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 2.0,
    },
    "Depth5_Long_LowLR": {
        "n_estimators": 3000,
        "learning_rate": 0.018,
        "max_depth": 5,
        "min_child_weight": 5,
        "subsample": 0.88,
        "colsample_bytree": 0.88,
        "reg_alpha": 0.02,
        "reg_lambda": 3.0,
    },
    "Depth4_Long": {
        "n_estimators": 3200,
        "learning_rate": 0.018,
        "max_depth": 4,
        "min_child_weight": 8,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.02,
        "reg_lambda": 4.0,
    },
    "Depth6_Regularized": {
        "n_estimators": 2200,
        "learning_rate": 0.020,
        "max_depth": 6,
        "min_child_weight": 12,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "gamma": 0.01,
        "reg_alpha": 0.05,
        "reg_lambda": 5.0,
    },
    "Lossguide_31": {
        "n_estimators": 3000,
        "learning_rate": 0.018,
        "max_depth": 0,
        "max_leaves": 31,
        "grow_policy": "lossguide",
        "min_child_weight": 8,
        "subsample": 0.88,
        "colsample_bytree": 0.88,
        "reg_alpha": 0.02,
        "reg_lambda": 4.0,
    },
}

rows = []
tuning_predictions = {}
holdout_predictions = {}
for name, parameters in candidate_parameters.items():
    print(f"Training {name}", flush=True)
    model = XGBClassifier(**common, **parameters)
    start = perf_counter()
    model.fit(
        train_matrix,
        train_target,
        eval_set=[(tuning_matrix, tuning_target)],
        verbose=False,
    )
    tuning_probability = model.predict_proba(tuning_matrix)[:, 1]
    holdout_probability = model.predict_proba(holdout_matrix)[:, 1]
    tuning_predictions[name] = tuning_probability
    holdout_predictions[name] = holdout_probability
    row = {
        "Model": name,
        "Best_Iteration": model.best_iteration,
        "Fit_Seconds": perf_counter() - start,
        **{f"Tuning_{key}": value for key, value in metrics(tuning_target, tuning_probability).items()},
        **{f"Holdout_{key}": value for key, value in metrics(holdout_target, holdout_probability).items()},
    }
    rows.append(row)
    print(row, flush=True)

model_names = list(tuning_predictions)
rank_tuning = {
    name: percentile_rank(tuning_predictions[name]) for name in model_names
}
rank_holdout = {
    name: percentile_rank(holdout_predictions[name]) for name in model_names
}
best_base = max(
    model_names,
    key=lambda name: roc_auc_score(tuning_target, tuning_predictions[name]),
)
weights = {name: 0.0 for name in model_names}
weights[best_base] = 1.0
best_tuning_auc = roc_auc_score(tuning_target, rank_tuning[best_base])

for _ in range(3):
    selected_weights = weights.copy()
    selected_auc = best_tuning_auc
    for candidate_name in model_names:
        for candidate_weight in np.linspace(0.01, 0.30, 30):
            trial_weights = {
                name: (1.0 - candidate_weight) * weight
                for name, weight in weights.items()
            }
            trial_weights[candidate_name] += candidate_weight
            tuning_blend = sum(
                trial_weights[name] * rank_tuning[name] for name in model_names
            )
            tuning_auc = roc_auc_score(tuning_target, tuning_blend)
            if tuning_auc > selected_auc:
                selected_auc = tuning_auc
                selected_weights = trial_weights
    if selected_auc <= best_tuning_auc + 1e-7:
        break
    weights = selected_weights
    best_tuning_auc = selected_auc

holdout_blend = sum(weights[name] * rank_holdout[name] for name in model_names)
rows.append({
    "Model": "Rank_Forward_Ensemble",
    "Best_Iteration": None,
    "Fit_Seconds": None,
    "Tuning_ROC_AUC": best_tuning_auc,
    "Tuning_Log_Loss": None,
    "Tuning_Average_Precision": average_precision_score(tuning_target, sum(weights[name] * rank_tuning[name] for name in model_names)),
    "Holdout_ROC_AUC": roc_auc_score(holdout_target, holdout_blend),
    "Holdout_Log_Loss": None,
    "Holdout_Average_Precision": average_precision_score(holdout_target, holdout_blend),
})

results = pd.DataFrame(rows).sort_values("Holdout_ROC_AUC", ascending=False)
results.to_csv(ARTIFACT_DIR / "main_xgboost_v2_results.csv", index=False)

summary = {
    "best_tuning_model": best_base,
    "ensemble_weights": {name: float(weight) for name, weight in weights.items()},
    "ensemble_tuning_auc": float(best_tuning_auc),
    "ensemble_holdout_auc": float(roc_auc_score(holdout_target, holdout_blend)),
}
with open(ARTIFACT_DIR / "main_xgboost_v2_summary.json", "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)

print(results.to_string(index=False), flush=True)
print(json.dumps(summary, indent=2), flush=True)
