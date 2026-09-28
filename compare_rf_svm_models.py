import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier, XGBRFClassifier


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


def metrics(target, scores):
    return {
        "ROC_AUC": roc_auc_score(target, scores),
        "Average_Precision": average_precision_score(target, scores),
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
print("Encoding features", flush=True)
train_matrix = preprocessor.fit_transform(train_features)
tuning_matrix = preprocessor.transform(tuning_features)
holdout_matrix = preprocessor.transform(holdout_features)

results = []
tuning_predictions = {}
holdout_predictions = {}


def record_result(name, model, fit_seconds, tuning_scores, holdout_scores):
    tuning_predictions[name] = tuning_scores
    holdout_predictions[name] = holdout_scores
    row = {
        "Model": name,
        "Fit_Seconds": fit_seconds,
        **{f"Tuning_{key}": value for key, value in metrics(tuning_target, tuning_scores).items()},
        **{f"Holdout_{key}": value for key, value in metrics(holdout_target, holdout_scores).items()},
    }
    results.append(row)
    print(row, flush=True)


print("Training XGBoost fine-bin reference", flush=True)
model = XGBClassifier(
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
)
start = perf_counter()
model.fit(train_matrix, train_target, eval_set=[(tuning_matrix, tuning_target)], verbose=False)
record_result(
    "XGBoost_FineBin_Reference",
    model,
    perf_counter() - start,
    model.predict_proba(tuning_matrix)[:, 1],
    model.predict_proba(holdout_matrix)[:, 1],
)

print("Training XGBoost random forest", flush=True)
model = XGBRFClassifier(
    n_estimators=350,
    learning_rate=1.0,
    max_depth=8,
    min_child_weight=10,
    subsample=0.80,
    colsample_bynode=0.80,
    reg_lambda=4.0,
    max_bin=1024,
    tree_method="hist",
    eval_metric="auc",
    n_jobs=4,
    random_state=RANDOM_STATE,
)
start = perf_counter()
model.fit(train_matrix, train_target, verbose=False)
record_result(
    "XGBoost_RandomForest",
    model,
    perf_counter() - start,
    model.predict_proba(tuning_matrix)[:, 1],
    model.predict_proba(holdout_matrix)[:, 1],
)

print("Training sklearn random forest", flush=True)
model = RandomForestClassifier(
    n_estimators=160,
    criterion="log_loss",
    max_depth=22,
    min_samples_leaf=10,
    max_features="sqrt",
    bootstrap=True,
    max_samples=0.80,
    n_jobs=1,
    random_state=RANDOM_STATE,
)
start = perf_counter()
model.fit(train_matrix, train_target)
record_result(
    "Sklearn_RandomForest",
    model,
    perf_counter() - start,
    model.predict_proba(tuning_matrix)[:, 1],
    model.predict_proba(holdout_matrix)[:, 1],
)

print("Scaling features for linear SVM", flush=True)
scaler = MaxAbsScaler()
svm_train_matrix = scaler.fit_transform(train_matrix)
svm_tuning_matrix = scaler.transform(tuning_matrix)
svm_holdout_matrix = scaler.transform(holdout_matrix)

for regularization in [0.01, 0.1, 1.0]:
    name = f"LinearSVM_C_{regularization:g}"
    print(f"Training {name}", flush=True)
    model = LinearSVC(
        C=regularization,
        class_weight=None,
        dual="auto",
        max_iter=5000,
        tol=1e-5,
        random_state=RANDOM_STATE,
    )
    start = perf_counter()
    model.fit(svm_train_matrix, train_target)
    record_result(
        name,
        model,
        perf_counter() - start,
        model.decision_function(svm_tuning_matrix),
        model.decision_function(svm_holdout_matrix),
    )

model_names = list(tuning_predictions)
rank_tuning = {
    name: percentile_rank(tuning_predictions[name]) for name in model_names
}
rank_holdout = {
    name: percentile_rank(holdout_predictions[name]) for name in model_names
}
base_name = "XGBoost_FineBin_Reference"
ensemble_rows = []
for candidate_name in model_names:
    if candidate_name == base_name:
        continue
    best_weight = 0.0
    best_tuning_auc = roc_auc_score(tuning_target, rank_tuning[base_name])
    for candidate_weight in np.linspace(0.01, 0.30, 30):
        blend = (
            (1.0 - candidate_weight) * rank_tuning[base_name]
            + candidate_weight * rank_tuning[candidate_name]
        )
        score = roc_auc_score(tuning_target, blend)
        if score > best_tuning_auc:
            best_tuning_auc = score
            best_weight = float(candidate_weight)
    holdout_blend = (
        (1.0 - best_weight) * rank_holdout[base_name]
        + best_weight * rank_holdout[candidate_name]
    )
    ensemble_rows.append({
        "Candidate": candidate_name,
        "Candidate_Weight": best_weight,
        "Tuning_ROC_AUC": best_tuning_auc,
        "Holdout_ROC_AUC": roc_auc_score(holdout_target, holdout_blend),
    })

results_frame = pd.DataFrame(results).sort_values("Holdout_ROC_AUC", ascending=False)
ensemble_frame = pd.DataFrame(ensemble_rows).sort_values("Holdout_ROC_AUC", ascending=False)
results_frame.to_csv(ARTIFACT_DIR / "rf_svm_model_comparison.csv", index=False)
ensemble_frame.to_csv(ARTIFACT_DIR / "rf_svm_pairwise_ensembles.csv", index=False)

summary = {
    "best_model": results_frame.iloc[0]["Model"],
    "best_model_holdout_auc": float(results_frame.iloc[0]["Holdout_ROC_AUC"]),
    "best_pairwise_candidate": ensemble_frame.iloc[0]["Candidate"],
    "best_pairwise_weight": float(ensemble_frame.iloc[0]["Candidate_Weight"]),
    "best_pairwise_holdout_auc": float(ensemble_frame.iloc[0]["Holdout_ROC_AUC"]),
}
with open(ARTIFACT_DIR / "rf_svm_experiment_summary.json", "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)

print(results_frame.to_string(index=False), flush=True)
print(ensemble_frame.to_string(index=False), flush=True)
print(json.dumps(summary, indent=2), flush=True)
