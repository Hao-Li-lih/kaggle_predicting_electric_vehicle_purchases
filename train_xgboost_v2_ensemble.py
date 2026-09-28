import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SUBMISSION_DIR = ROOT / "submissions"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "Will_Buy_EV"
ID_COLUMN = "id"


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


def percentile_rank(values):
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def save_submission(filename, ids, predictions, sample_submission):
    submission = pd.DataFrame({ID_COLUMN: ids, TARGET: predictions})
    assert submission.shape == sample_submission.shape
    assert submission[ID_COLUMN].equals(sample_submission[ID_COLUMN])
    assert submission[TARGET].notna().all()
    assert submission[TARGET].between(0, 1).all()
    assert submission[TARGET].nunique() > 1
    path = SUBMISSION_DIR / filename
    submission.to_csv(path, index=False)
    print(f"Saved {path.name}: mean={submission[TARGET].mean():.6f}", flush=True)


SUBMISSION_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)
train_data = pd.read_csv(DATA_DIR / "train.csv")
test_data = pd.read_csv(DATA_DIR / "test.csv")
sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")
target = train_data[TARGET].eq("Yes").astype(int)

print("Engineering and encoding features", flush=True)
train_features = engineer_features(train_data.drop(columns=[TARGET]))
test_features = engineer_features(test_data)
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
test_matrix = preprocessor.transform(test_features)

base_parameters = {
    "n_estimators": 2160,
    "learning_rate": 0.025,
    "max_depth": 5,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 2.0,
    "max_bin": 4096,
    "tree_method": "hist",
    "eval_metric": "auc",
    "n_jobs": 4,
}

training_rows = []
base_predictions = []
for seed in [42, 314, 2026, 2718, 8675]:
    print(f"Training extended depth-5 seed={seed}", flush=True)
    model = XGBClassifier(**base_parameters, random_state=seed)
    start = perf_counter()
    model.fit(train_matrix, target, verbose=False)
    predictions = model.predict_proba(test_matrix)[:, 1]
    base_predictions.append(predictions)
    training_rows.append({
        "Model": "Depth5_Extended",
        "Seed": seed,
        "Fit_Seconds": perf_counter() - start,
    })
    save_submission(
        f"submission_xgb4096_extended_seed_{seed}.csv",
        test_data[ID_COLUMN],
        predictions,
        sample_submission,
    )

base_ensemble = np.mean(base_predictions, axis=0)
save_submission(
    "submission_xgb4096_extended_5seed_ensemble.csv",
    test_data[ID_COLUMN],
    base_ensemble,
    sample_submission,
)

print("Training low-learning-rate structural model", flush=True)
low_lr_model = XGBClassifier(
    n_estimators=2987,
    learning_rate=0.018,
    max_depth=5,
    min_child_weight=5,
    subsample=0.88,
    colsample_bytree=0.88,
    reg_alpha=0.02,
    reg_lambda=3.0,
    max_bin=4096,
    tree_method="hist",
    eval_metric="auc",
    n_jobs=4,
    random_state=42,
)
start = perf_counter()
low_lr_model.fit(train_matrix, target, verbose=False)
low_lr_predictions = low_lr_model.predict_proba(test_matrix)[:, 1]
training_rows.append({
    "Model": "Depth5_Long_LowLR",
    "Seed": 42,
    "Fit_Seconds": perf_counter() - start,
})

print("Training lossguide structural model", flush=True)
lossguide_model = XGBClassifier(
    n_estimators=2314,
    learning_rate=0.018,
    max_depth=0,
    max_leaves=31,
    grow_policy="lossguide",
    min_child_weight=8,
    subsample=0.88,
    colsample_bytree=0.88,
    reg_alpha=0.02,
    reg_lambda=4.0,
    max_bin=4096,
    tree_method="hist",
    eval_metric="auc",
    n_jobs=4,
    random_state=42,
)
start = perf_counter()
lossguide_model.fit(train_matrix, target, verbose=False)
lossguide_predictions = lossguide_model.predict_proba(test_matrix)[:, 1]
training_rows.append({
    "Model": "Lossguide_31",
    "Seed": 42,
    "Fit_Seconds": perf_counter() - start,
})

rank_ensemble = (
    0.623 * percentile_rank(base_ensemble)
    + 0.267 * percentile_rank(low_lr_predictions)
    + 0.110 * percentile_rank(lossguide_predictions)
)
save_submission(
    "submission_xgb_v2_structural_rank_ensemble.csv",
    test_data[ID_COLUMN],
    rank_ensemble,
    sample_submission,
)

finebin_8192 = pd.read_csv(
    SUBMISSION_DIR / "submission_xgboost_finebin_8192_seed_42.csv"
)[TARGET].to_numpy()
public_feedback_blend = 0.80 * base_ensemble + 0.20 * finebin_8192
save_submission(
    "submission_xgb_v2_public_blend_80_20.csv",
    test_data[ID_COLUMN],
    public_feedback_blend,
    sample_submission,
)

pd.DataFrame(training_rows).to_csv(
    ARTIFACT_DIR / "xgboost_v2_training_summary.csv",
    index=False,
)
