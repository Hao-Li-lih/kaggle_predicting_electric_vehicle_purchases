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
    print(
        f"Saved {path.name}: mean={submission[TARGET].mean():.6f}, "
        f"min={submission[TARGET].min():.6f}, "
        f"max={submission[TARGET].max():.6f}",
        flush=True,
    )
    return path


SUBMISSION_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)

print("Loading data", flush=True)
train_data = pd.read_csv(DATA_DIR / "train.csv")
test_data = pd.read_csv(DATA_DIR / "test.csv")
sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")
target = train_data[TARGET].eq("Yes").astype(int)
train_features = engineer_features(train_data.drop(columns=[TARGET]))
test_features = engineer_features(test_data)

print("Encoding features", flush=True)
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
    "n_estimators": 1800,
    "learning_rate": 0.025,
    "max_depth": 5,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 2.0,
    "tree_method": "hist",
    "eval_metric": "auc",
    "n_jobs": 4,
}

training_rows = []
finebin_4096_predictions = []
for seed in [42, 314]:
    print(f"Training max_bin=4096, seed={seed}", flush=True)
    model = XGBClassifier(
        **base_parameters,
        max_bin=4096,
        random_state=seed,
    )
    start = perf_counter()
    model.fit(train_matrix, target, verbose=False)
    predictions = model.predict_proba(test_matrix)[:, 1]
    elapsed = perf_counter() - start
    finebin_4096_predictions.append(predictions)
    training_rows.append(
        {"Model": "XGBoost", "Max_Bin": 4096, "Seed": seed, "Fit_Seconds": elapsed}
    )
    save_submission(
        f"submission_xgboost_finebin_4096_seed_{seed}.csv",
        test_data[ID_COLUMN],
        predictions,
        sample_submission,
    )

finebin_4096_ensemble = np.mean(finebin_4096_predictions, axis=0)
save_submission(
    "submission_xgboost_finebin_4096_seed_ensemble.csv",
    test_data[ID_COLUMN],
    finebin_4096_ensemble,
    sample_submission,
)

print("Training max_bin=8192, seed=42", flush=True)
finebin_8192_model = XGBClassifier(
    **base_parameters,
    max_bin=8192,
    random_state=42,
)
start = perf_counter()
finebin_8192_model.fit(train_matrix, target, verbose=False)
finebin_8192_predictions = finebin_8192_model.predict_proba(test_matrix)[:, 1]
training_rows.append(
    {
        "Model": "XGBoost",
        "Max_Bin": 8192,
        "Seed": 42,
        "Fit_Seconds": perf_counter() - start,
    }
)
save_submission(
    "submission_xgboost_finebin_8192_seed_42.csv",
    test_data[ID_COLUMN],
    finebin_8192_predictions,
    sample_submission,
)

probability_blend = (
    0.40 * finebin_4096_ensemble
    + 0.60 * finebin_8192_predictions
)
save_submission(
    "submission_xgboost_finebin_probability_blend.csv",
    test_data[ID_COLUMN],
    probability_blend,
    sample_submission,
)

rank_blend = (
    0.40 * percentile_rank(finebin_4096_ensemble)
    + 0.60 * percentile_rank(finebin_8192_predictions)
)
save_submission(
    "submission_xgboost_finebin_rank_blend.csv",
    test_data[ID_COLUMN],
    rank_blend,
    sample_submission,
)

prediction_summary = pd.DataFrame({
    "Submission": [
        "finebin_4096_ensemble",
        "finebin_8192_seed_42",
        "finebin_probability_blend",
        "finebin_rank_blend",
    ],
    "Mean": [
        finebin_4096_ensemble.mean(),
        finebin_8192_predictions.mean(),
        probability_blend.mean(),
        rank_blend.mean(),
    ],
    "Standard_Deviation": [
        finebin_4096_ensemble.std(),
        finebin_8192_predictions.std(),
        probability_blend.std(),
        rank_blend.std(),
    ],
})
prediction_summary.to_csv(
    ARTIFACT_DIR / "finebin_submission_summary.csv",
    index=False,
)
pd.DataFrame(training_rows).to_csv(
    ARTIFACT_DIR / "finebin_training_summary.csv",
    index=False,
)
print(prediction_summary.to_string(index=False), flush=True)
