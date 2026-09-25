import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
TARGET = "Will_Buy_EV"


def prepare(data):
    frame = data.copy().drop(columns=["id", "Buyer_ID"], errors="ignore")
    frame["Total_Charging_Stations"] = (
        frame["Charging_Stations_Near_Home"]
        + frame["Charging_Stations_Near_Work"]
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
    return frame


competition = pd.read_csv(ROOT / "data" / "train.csv")
original = pd.read_csv(
    ROOT / "external_data" / "original" / "EV_Adoption_and_Range_Anxiety_Dataset.csv"
)
competition_train, competition_holdout = train_test_split(
    competition,
    test_size=0.15,
    random_state=42,
    stratify=competition[TARGET],
)

original_features = prepare(original.drop(columns=[TARGET]))
competition_train_features = prepare(competition_train.drop(columns=[TARGET]))
competition_holdout_features = prepare(competition_holdout.drop(columns=[TARGET]))
all_features = pd.concat(
    [original_features, competition_train_features, competition_holdout_features],
    ignore_index=True,
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
preprocessor.fit(all_features)
original_matrix = preprocessor.transform(original_features)
competition_train_matrix = preprocessor.transform(competition_train_features)
competition_holdout_matrix = preprocessor.transform(competition_holdout_features)

original_target = original[TARGET].eq("Yes").astype(int)
competition_train_target = competition_train[TARGET].eq("Yes").astype(int)
competition_holdout_target = competition_holdout[TARGET].eq("Yes").astype(int)

parameters = {
    "n_estimators": 800,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 3.0,
    "tree_method": "hist",
    "eval_metric": "auc",
    "n_jobs": 4,
    "random_state": 42,
}

rows = []
for name, matrix, target in [
    ("original_only", original_matrix, original_target),
    ("competition_only", competition_train_matrix, competition_train_target),
]:
    start = perf_counter()
    model = XGBClassifier(**parameters)
    model.fit(matrix, target, verbose=False)
    probabilities = model.predict_proba(competition_holdout_matrix)[:, 1]
    result = {
        "Training_Source": name,
        "Fit_Seconds": perf_counter() - start,
        "Holdout_ROC_AUC": roc_auc_score(competition_holdout_target, probabilities),
        "Holdout_Log_Loss": log_loss(competition_holdout_target, probabilities),
    }
    rows.append(result)
    print(result, flush=True)

pd.DataFrame(rows).to_csv(
    ROOT / "artifacts" / "original_prior_probe_results.csv",
    index=False,
)
