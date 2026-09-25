import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


ROOT = Path(__file__).resolve().parent
TARGET = "Will_Buy_EV"
ID_COLUMN = "id"
RANDOM_STATE = 42


def prepare_features(data):
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
    frame["Charging_Station_Gap"] = (
        frame["Charging_Stations_Near_Home"]
        - frame["Charging_Stations_Near_Work"]
    )

    concern = frame["Environmental_Concern_Level"].astype(int).astype(str)
    cars = frame["Number_of_Cars_Owned"].astype(str)
    home_stations = frame["Charging_Stations_Near_Home"].astype(str)
    work_stations = frame["Charging_Stations_Near_Work"].astype(str)
    frame["Environmental_Concern_Category"] = concern
    frame["Number_of_Cars_Category"] = cars
    frame["Home_Stations_Category"] = home_stations
    frame["Work_Stations_Category"] = work_stations
    frame["Core_Context"] = (
        concern
        + "_"
        + frame["Subsidy_Available"].astype(str)
        + "_"
        + frame["Range_Anxiety_Level"].astype(str)
        + "_"
        + frame["Home_Charging_Possible"].astype(str)
    )
    frame["Concern_Subsidy_Context"] = (
        concern + "_" + frame["Subsidy_Available"].astype(str)
    )
    frame["Concern_Anxiety_Context"] = (
        concern + "_" + frame["Range_Anxiety_Level"].astype(str)
    )
    frame["Income_Context"] = (
        frame["Subsidy_Available"].astype(str)
        + "_"
        + frame["Range_Anxiety_Level"].astype(str)
    )
    frame["City_Charging_Context"] = (
        frame["City_Type"].astype(str)
        + "_"
        + frame["Home_Charging_Possible"].astype(str)
    )
    frame["Car_City_Context"] = (
        frame["Current_Car_Type"].astype(str)
        + "_"
        + frame["City_Type"].astype(str)
    )
    return frame


def build_preprocessor(frame, spline_knots):
    spline_columns = ["Age", "Annual_Income_USD", "Daily_Commute_km"]
    categorical_columns = [
        "Gender",
        "City_Type",
        "Current_Car_Type",
        "Home_Charging_Possible",
        "Subsidy_Available",
        "Range_Anxiety_Level",
        "Environmental_Concern_Category",
        "Number_of_Cars_Category",
        "Home_Stations_Category",
        "Work_Stations_Category",
        "Core_Context",
        "Concern_Subsidy_Context",
        "Concern_Anxiety_Context",
        "Income_Context",
        "City_Charging_Context",
        "Car_City_Context",
    ]
    numeric_columns = [
        "Total_Charging_Stations",
        "Min_Charging_Stations",
        "Max_Charging_Stations",
        "Charging_Station_Gap",
    ]
    spline_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "spline",
            SplineTransformer(
                n_knots=spline_knots,
                degree=3,
                knots="quantile",
                include_bias=False,
            ),
        ),
        ("scaler", StandardScaler()),
    ])
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("spline", spline_pipeline, spline_columns),
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])


def calculate_metrics(target, probabilities):
    return {
        "ROC_AUC": roc_auc_score(target, probabilities),
        "Log_Loss": log_loss(target, probabilities),
        "Average_Precision": average_precision_score(target, probabilities),
    }


data = pd.read_csv(ROOT / "data" / "train.csv")
features = prepare_features(data.drop(columns=[TARGET]))
target = data[TARGET].eq("Yes").astype(int)
train_features, holdout_features, train_target, holdout_target = train_test_split(
    features,
    target,
    test_size=0.15,
    random_state=RANDOM_STATE,
    stratify=target,
)

rows = []
for spline_knots in [5, 8, 12]:
    for regularization in [0.1, 1.0, 10.0]:
        preprocessor = build_preprocessor(train_features, spline_knots)
        model = LogisticRegression(
            C=regularization,
            max_iter=300,
            solver="lbfgs",
            tol=1e-7,
        )
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        start = perf_counter()
        pipeline.fit(train_features, train_target)
        probabilities = pipeline.predict_proba(holdout_features)[:, 1]
        result = {
            "Spline_Knots": spline_knots,
            "C": regularization,
            "Fit_Seconds": perf_counter() - start,
            **calculate_metrics(holdout_target, probabilities),
        }
        rows.append(result)
        print(result, flush=True)

results = pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False)
print(results.to_string(index=False), flush=True)
results.to_csv(ROOT / "artifacts" / "structured_model_probe_results.csv", index=False)
