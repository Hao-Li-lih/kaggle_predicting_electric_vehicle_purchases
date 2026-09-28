import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.ensemble import ExtraTreesClassifier
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
    frame["Total_Charging_Stations"] = (
        frame["Charging_Stations_Near_Home"]
        + frame["Charging_Stations_Near_Work"]
    )
    frame["Min_Charging_Stations"] = frame[
        ["Charging_Stations_Near_Home", "Charging_Stations_Near_Work"]
    ].min(axis=1)
    frame["Max_Charging_Stations"] = frame[
        ["Charging_Stations_Near_Home", "Charging_Stations_Near_Work"]
    ].max(axis=1)
    frame["Home_Work_Station_Gap"] = (
        frame["Charging_Stations_Near_Home"]
        - frame["Charging_Stations_Near_Work"]
    )
    frame["Commute_Per_Station"] = (
        frame["Daily_Commute_km"] / (frame["Total_Charging_Stations"] + 1.0)
    )
    frame["Log_Annual_Income"] = np.log1p(frame["Annual_Income_USD"])
    frame["Income_Per_Car"] = (
        frame["Annual_Income_USD"] / frame["Number_of_Cars_Owned"].clip(lower=1)
    )
    frame["Income_Per_Commute"] = (
        frame["Annual_Income_USD"] / frame["Daily_Commute_km"].clip(lower=1)
    )
    frame["Environment_Income_Interaction"] = (
        frame["Environmental_Concern_Level"] * frame["Log_Annual_Income"]
    )
    frame["Environment_Charging_Interaction"] = (
        frame["Environmental_Concern_Level"] * frame["Total_Charging_Stations"]
    )

    home_flag = frame["Home_Charging_Possible"].map({"No": 0.0, "Yes": 1.0})
    subsidy_flag = frame["Subsidy_Available"].map({"No": 0.0, "Yes": 1.0})
    anxiety = frame["Range_Anxiety_Level"].map(
        {"Low": 0.0, "Medium": 1.0, "High": 2.0}
    )
    frame["Home_Charging_Flag"] = home_flag
    frame["Subsidy_Flag"] = subsidy_flag
    frame["Range_Anxiety_Ordinal"] = anxiety
    frame["Age_Squared"] = frame["Age"] ** 2
    frame["Income_Squared_Scaled"] = (
        frame["Annual_Income_USD"] / 100_000.0
    ) ** 2
    frame["Log_Daily_Commute"] = np.log1p(frame["Daily_Commute_km"])
    frame["Charging_Balance_Ratio"] = (
        (frame["Charging_Stations_Near_Home"] + 1.0)
        / (frame["Charging_Stations_Near_Work"] + 1.0)
    )
    frame["Income_x_Subsidy"] = frame["Log_Annual_Income"] * subsidy_flag
    frame["Income_x_Home_Charging"] = frame["Log_Annual_Income"] * home_flag
    frame["Concern_x_Subsidy"] = (
        frame["Environmental_Concern_Level"] * subsidy_flag
    )
    frame["Concern_x_Home_Charging"] = (
        frame["Environmental_Concern_Level"] * home_flag
    )
    frame["Anxiety_x_Commute"] = anxiety * frame["Daily_Commute_km"]
    frame["Anxiety_x_Charging"] = anxiety * frame["Total_Charging_Stations"]
    frame["Concern_Minus_Anxiety"] = (
        frame["Environmental_Concern_Level"] - anxiety
    )
    frame["Core_Context"] = (
        frame["Environmental_Concern_Level"].astype(str)
        + "_"
        + frame["Subsidy_Available"].astype(str)
        + "_"
        + frame["Range_Anxiety_Level"].astype(str)
        + "_"
        + frame["Home_Charging_Possible"].astype(str)
    )
    return frame.replace([np.inf, -np.inf], np.nan)


def calculate_metrics(target, probabilities):
    return {
        "ROC_AUC": roc_auc_score(target, probabilities),
        "Log_Loss": log_loss(target, probabilities),
        "Average_Precision": average_precision_score(target, probabilities),
    }


def percentile_rank(values):
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def softmax(values):
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


ARTIFACT_DIR.mkdir(exist_ok=True)
print("Loading data", flush=True)
data = pd.read_csv(DATA_DIR / "train.csv")
target = data[TARGET].eq("Yes").astype(int)
features = engineer_features(data.drop(columns=[TARGET]))

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
print("Encoding tree matrices", flush=True)
train_matrix = preprocessor.fit_transform(train_features)
tuning_matrix = preprocessor.transform(tuning_features)
holdout_matrix = preprocessor.transform(holdout_features)

model_specs = {
    "XGBoost_FineBin_Depth5": XGBClassifier(
        n_estimators=1800,
        learning_rate=0.025,
        max_depth=5,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        max_bin=4096,
        tree_method="hist",
        eval_metric="auc",
        early_stopping_rounds=100,
        n_jobs=4,
        random_state=RANDOM_STATE,
    ),
    "XGBoost_FineBin_Depth3": XGBClassifier(
        n_estimators=2800,
        learning_rate=0.02,
        max_depth=3,
        min_child_weight=10,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=4.0,
        max_bin=4096,
        tree_method="hist",
        eval_metric="auc",
        early_stopping_rounds=120,
        n_jobs=4,
        random_state=RANDOM_STATE,
    ),
    "LightGBM_GOSS": LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=100,
        subsample=1.0,
        colsample_bytree=0.90,
        reg_alpha=0.05,
        reg_lambda=4.0,
        max_bin=1023,
        data_sample_strategy="goss",
        top_rate=0.20,
        other_rate=0.10,
        objective="binary",
        n_jobs=4,
        random_state=RANDOM_STATE,
        verbosity=-1,
    ),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=60,
        criterion="entropy",
        max_features=0.60,
        min_samples_leaf=5,
        class_weight=None,
        n_jobs=1,
        random_state=RANDOM_STATE,
    ),
}

rows = []
tuning_predictions = {}
holdout_predictions = {}
for name, model in model_specs.items():
    print(f"Training {name}", flush=True)
    start = perf_counter()
    if name.startswith("XGBoost"):
        model.fit(
            train_matrix,
            train_target,
            eval_set=[(tuning_matrix, tuning_target)],
            verbose=False,
        )
        best_iteration = model.best_iteration
    elif name == "LightGBM_GOSS":
        model.fit(
            train_matrix,
            train_target,
            eval_set=[(tuning_matrix, tuning_target)],
            eval_metric="auc",
            callbacks=[early_stopping(120, verbose=False), log_evaluation(0)],
        )
        best_iteration = model.best_iteration_
    else:
        model.fit(train_matrix, train_target)
        best_iteration = None
    tuning_probability = model.predict_proba(tuning_matrix)[:, 1]
    holdout_probability = model.predict_proba(holdout_matrix)[:, 1]
    tuning_predictions[name] = tuning_probability
    holdout_predictions[name] = holdout_probability
    result = {
        "Model": name,
        "Best_Iteration": best_iteration,
        "Fit_Seconds": perf_counter() - start,
        **{f"Tuning_{key}": value for key, value in calculate_metrics(tuning_target, tuning_probability).items()},
        **{f"Holdout_{key}": value for key, value in calculate_metrics(holdout_target, holdout_probability).items()},
    }
    rows.append(result)
    print(result, flush=True)

categorical_columns = train_features.select_dtypes(exclude=np.number).columns.tolist()
catboost_train = train_features.copy()
catboost_tuning = tuning_features.copy()
catboost_holdout = holdout_features.copy()
for column in categorical_columns:
    catboost_train[column] = catboost_train[column].fillna("Missing").astype(str)
    catboost_tuning[column] = catboost_tuning[column].fillna("Missing").astype(str)
    catboost_holdout[column] = catboost_holdout[column].fillna("Missing").astype(str)

print("Training CatBoost_Lossguide", flush=True)
catboost_model = CatBoostClassifier(
    iterations=2200,
    learning_rate=0.03,
    grow_policy="Lossguide",
    max_leaves=31,
    min_data_in_leaf=100,
    l2_leaf_reg=5.0,
    random_strength=0.5,
    bootstrap_type="Bernoulli",
    subsample=0.85,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=RANDOM_STATE,
    thread_count=4,
    od_type="Iter",
    od_wait=120,
    verbose=False,
)
start = perf_counter()
catboost_model.fit(
    catboost_train,
    train_target,
    cat_features=categorical_columns,
    eval_set=(catboost_tuning, tuning_target),
    use_best_model=True,
)
catboost_tuning_probability = catboost_model.predict_proba(catboost_tuning)[:, 1]
catboost_holdout_probability = catboost_model.predict_proba(catboost_holdout)[:, 1]
tuning_predictions["CatBoost_Lossguide"] = catboost_tuning_probability
holdout_predictions["CatBoost_Lossguide"] = catboost_holdout_probability
catboost_result = {
    "Model": "CatBoost_Lossguide",
    "Best_Iteration": catboost_model.get_best_iteration(),
    "Fit_Seconds": perf_counter() - start,
    **{f"Tuning_{key}": value for key, value in calculate_metrics(tuning_target, catboost_tuning_probability).items()},
    **{f"Holdout_{key}": value for key, value in calculate_metrics(holdout_target, catboost_holdout_probability).items()},
}
rows.append(catboost_result)
print(catboost_result, flush=True)

model_names = list(tuning_predictions)
tuning_matrix_predictions = np.column_stack([tuning_predictions[name] for name in model_names])
holdout_matrix_predictions = np.column_stack([holdout_predictions[name] for name in model_names])

def objective(logits):
    weights = softmax(logits)
    blended = tuning_matrix_predictions @ weights
    return log_loss(tuning_target, blended)


optimization = minimize(
    objective,
    x0=np.zeros(len(model_names)),
    method="BFGS",
    options={"maxiter": 300},
)
logloss_weights = softmax(optimization.x)
logloss_tuning_blend = tuning_matrix_predictions @ logloss_weights
logloss_holdout_blend = holdout_matrix_predictions @ logloss_weights

rank_tuning_matrix = np.column_stack([
    percentile_rank(tuning_predictions[name]) for name in model_names
])
rank_holdout_matrix = np.column_stack([
    percentile_rank(holdout_predictions[name]) for name in model_names
])
best_index = int(np.argmax([roc_auc_score(tuning_target, tuning_predictions[name]) for name in model_names]))
rank_weights = np.zeros(len(model_names))
rank_weights[best_index] = 1.0
best_rank_auc = roc_auc_score(tuning_target, rank_tuning_matrix @ rank_weights)

for _ in range(2):
    selected_weights = rank_weights.copy()
    selected_auc = best_rank_auc
    for candidate_index in range(len(model_names)):
        for candidate_weight in np.linspace(0.02, 0.30, 15):
            trial_weights = (1.0 - candidate_weight) * rank_weights
            trial_weights[candidate_index] += candidate_weight
            trial_prediction = rank_tuning_matrix @ trial_weights
            trial_auc = roc_auc_score(tuning_target, trial_prediction)
            if trial_auc > selected_auc:
                selected_auc = trial_auc
                selected_weights = trial_weights
    if selected_auc <= best_rank_auc + 1e-7:
        break
    rank_weights = selected_weights
    best_rank_auc = selected_auc

rank_tuning_blend = rank_tuning_matrix @ rank_weights
rank_holdout_blend = rank_holdout_matrix @ rank_weights

for ensemble_name, tuning_probability, holdout_probability in [
    ("Ensemble_LogLoss_Optimized", logloss_tuning_blend, logloss_holdout_blend),
    ("Ensemble_Rank_Forward", rank_tuning_blend, rank_holdout_blend),
]:
    rows.append({
        "Model": ensemble_name,
        "Best_Iteration": None,
        "Fit_Seconds": None,
        **{f"Tuning_{key}": value for key, value in calculate_metrics(tuning_target, tuning_probability).items()},
        **{f"Holdout_{key}": value for key, value in calculate_metrics(holdout_target, holdout_probability).items()},
    })

results = pd.DataFrame(rows).sort_values("Holdout_ROC_AUC", ascending=False)
results.to_csv(ARTIFACT_DIR / "diverse_model_comparison.csv", index=False)

weights = {
    "models": model_names,
    "logloss_weights": {name: float(weight) for name, weight in zip(model_names, logloss_weights)},
    "rank_weights": {name: float(weight) for name, weight in zip(model_names, rank_weights)},
    "logloss_optimization_success": bool(optimization.success),
}
with open(ARTIFACT_DIR / "diverse_ensemble_weights.json", "w", encoding="utf-8") as file:
    json.dump(weights, file, indent=2)

print(results.to_string(index=False), flush=True)
print(json.dumps(weights, indent=2), flush=True)
