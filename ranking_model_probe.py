import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

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
from xgboost import XGBClassifier, XGBRanker


ROOT = Path(__file__).resolve().parent
TARGET = "Will_Buy_EV"
ID_COLUMN = "id"


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
    anxiety = frame["Range_Anxiety_Level"].map({"Low": 0.0, "Medium": 1.0, "High": 2.0})
    frame["Home_Charging_Flag"] = home_flag
    frame["Subsidy_Flag"] = subsidy_flag
    frame["Range_Anxiety_Ordinal"] = anxiety
    frame["Age_Squared"] = frame["Age"] ** 2
    frame["Income_Squared_Scaled"] = (frame["Annual_Income_USD"] / 100_000.0) ** 2
    frame["Log_Daily_Commute"] = np.log1p(frame["Daily_Commute_km"])
    frame["Charging_Balance_Ratio"] = (
        (frame["Charging_Stations_Near_Home"] + 1.0)
        / (frame["Charging_Stations_Near_Work"] + 1.0)
    )
    frame["Income_x_Subsidy"] = frame["Log_Annual_Income"] * subsidy_flag
    frame["Income_x_Home_Charging"] = frame["Log_Annual_Income"] * home_flag
    frame["Concern_x_Subsidy"] = frame["Environmental_Concern_Level"] * subsidy_flag
    frame["Concern_x_Home_Charging"] = frame["Environmental_Concern_Level"] * home_flag
    frame["Anxiety_x_Commute"] = anxiety * frame["Daily_Commute_km"]
    frame["Anxiety_x_Charging"] = anxiety * frame["Total_Charging_Stations"]
    frame["Concern_Minus_Anxiety"] = frame["Environmental_Concern_Level"] - anxiety
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


def calculate_metrics(target, scores, probabilities=None):
    result = {
        "ROC_AUC": roc_auc_score(target, scores),
        "Average_Precision": average_precision_score(target, scores),
    }
    if probabilities is not None:
        result["Log_Loss"] = log_loss(target, probabilities)
    return result


data = pd.read_csv(ROOT / "data" / "train.csv")
target = data[TARGET].eq("Yes").astype(int)
features = engineer_features(data.drop(columns=[TARGET]))
train_tune_features, holdout_features, train_tune_target, holdout_target = train_test_split(
    features,
    target,
    test_size=0.15,
    random_state=42,
    stratify=target,
)
train_features, tuning_features, train_target, tuning_target = train_test_split(
    train_tune_features,
    train_tune_target,
    test_size=0.17647058823529413,
    random_state=43,
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
train_matrix = preprocessor.fit_transform(train_features)
tuning_matrix = preprocessor.transform(tuning_features)
holdout_matrix = preprocessor.transform(holdout_features)

base_parameters = {
    "n_estimators": 1800,
    "learning_rate": 0.025,
    "max_depth": 5,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 2.0,
    "tree_method": "hist",
    "n_jobs": 4,
    "random_state": 42,
    "early_stopping_rounds": 100,
}

rows = [
    {
        "Model": "binary_max_bin_256",
        "Best_Iteration": 1511,
        "Fit_Seconds": 60.7464191,
        "Tuning_ROC_AUC": 0.9417009972,
        "Tuning_Average_Precision": 0.7563136847,
        "Tuning_Log_Loss": 0.2267530859,
        "Holdout_ROC_AUC": 0.9418317514,
        "Holdout_Average_Precision": 0.7582096197,
        "Holdout_Log_Loss": 0.2263486534,
    },
    {
        "Model": "binary_max_bin_512",
        "Best_Iteration": 1799,
        "Fit_Seconds": 73.6533205,
        "Tuning_ROC_AUC": 0.9421457555,
        "Tuning_Average_Precision": 0.7584814551,
        "Tuning_Log_Loss": 0.2259783149,
        "Holdout_ROC_AUC": 0.9421959504,
        "Holdout_Average_Precision": 0.7596232866,
        "Holdout_Log_Loss": 0.2257353067,
    },
    {
        "Model": "binary_max_bin_1024",
        "Best_Iteration": 1789,
        "Fit_Seconds": 77.2772679,
        "Tuning_ROC_AUC": 0.9423939302,
        "Tuning_Average_Precision": 0.7605419939,
        "Tuning_Log_Loss": 0.2254532874,
        "Holdout_ROC_AUC": 0.9425200796,
        "Holdout_Average_Precision": 0.7614164038,
        "Holdout_Log_Loss": 0.2251572162,
    },
    {
        "Model": "binary_max_bin_2048",
        "Best_Iteration": 1791,
        "Fit_Seconds": 74.5061970,
        "Tuning_ROC_AUC": 0.9427416863,
        "Tuning_Average_Precision": 0.7620507091,
        "Tuning_Log_Loss": 0.2248154730,
        "Holdout_ROC_AUC": 0.9428077402,
        "Holdout_Average_Precision": 0.7633901574,
        "Holdout_Log_Loss": 0.2246048003,
    },
    {
        "Model": "binary_max_bin_4096",
        "Best_Iteration": 1799,
        "Fit_Seconds": 93.6020434,
        "Tuning_ROC_AUC": 0.9429511018,
        "Tuning_Average_Precision": 0.7632595904,
        "Tuning_Log_Loss": 0.2244485617,
        "Holdout_ROC_AUC": 0.9429931638,
        "Holdout_Average_Precision": 0.7643908686,
        "Holdout_Log_Loss": 0.2242611051,
    },
]
predictions = {}
for max_bin in [8192]:
    model = XGBClassifier(**base_parameters, max_bin=max_bin, eval_metric="auc")
    start = perf_counter()
    model.fit(train_matrix, train_target, eval_set=[(tuning_matrix, tuning_target)], verbose=False)
    tuning_probabilities = model.predict_proba(tuning_matrix)[:, 1]
    holdout_probabilities = model.predict_proba(holdout_matrix)[:, 1]
    name = f"binary_max_bin_{max_bin}"
    predictions[name] = holdout_probabilities
    result = {
        "Model": name,
        "Best_Iteration": model.best_iteration,
        "Fit_Seconds": perf_counter() - start,
        **{f"Tuning_{key}": value for key, value in calculate_metrics(tuning_target, tuning_probabilities, tuning_probabilities).items()},
        **{f"Holdout_{key}": value for key, value in calculate_metrics(holdout_target, holdout_probabilities, holdout_probabilities).items()},
    }
    rows.append(result)
    print(result, flush=True)

results = pd.DataFrame(rows).sort_values("Holdout_ROC_AUC", ascending=False)
print(results.to_string(index=False), flush=True)
results.to_csv(ROOT / "artifacts" / "ranking_model_probe_results.csv", index=False)
