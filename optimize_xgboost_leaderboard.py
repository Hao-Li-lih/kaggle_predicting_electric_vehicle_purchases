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


RANDOM_STATE = 42
TARGET = "Will_Buy_EV"
ID_COLUMN = "id"
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ORIGINAL_PATH = ROOT / "external_data" / "original" / "EV_Adoption_and_Range_Anxiety_Dataset.csv"
SUBMISSION_DIR = ROOT / "submissions"
ARTIFACT_DIR = ROOT / "artifacts"
SUBMISSION_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)


def engineer_features(data, refined=True):
    frame = data.copy().drop(columns=[ID_COLUMN, "Buyer_ID"], errors="ignore")
    frame["Total_Charging_Stations"] = (
        frame["Charging_Stations_Near_Home"] + frame["Charging_Stations_Near_Work"]
    )
    frame["Min_Charging_Stations"] = frame[
        ["Charging_Stations_Near_Home", "Charging_Stations_Near_Work"]
    ].min(axis=1)
    frame["Max_Charging_Stations"] = frame[
        ["Charging_Stations_Near_Home", "Charging_Stations_Near_Work"]
    ].max(axis=1)
    frame["Home_Work_Station_Gap"] = (
        frame["Charging_Stations_Near_Home"] - frame["Charging_Stations_Near_Work"]
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
    frame["Age_Band"] = pd.cut(
        frame["Age"],
        [-np.inf, 29, 39, 49, 59, np.inf],
        labels=["25_29", "30_39", "40_49", "50_59", "60_plus"],
    ).astype(str)
    frame["Income_Band"] = pd.cut(
        frame["Annual_Income_USD"],
        [-np.inf, 50_000, 75_000, 100_000, 125_000, np.inf],
        labels=["under_50k", "50k_75k", "75k_100k", "100k_125k", "125k_plus"],
    ).astype(str)
    frame["Commute_Band"] = pd.cut(
        frame["Daily_Commute_km"],
        [-np.inf, 10, 25, 50, 75, np.inf],
        labels=["under_10", "10_25", "25_50", "50_75", "75_plus"],
    ).astype(str)
    frame["Charging_Context"] = (
        frame["Home_Charging_Possible"].astype(str)
        + "_"
        + frame["Subsidy_Available"].astype(str)
    )

    if refined:
        home_flag = frame["Home_Charging_Possible"].map({"No": 0.0, "Yes": 1.0})
        subsidy_flag = frame["Subsidy_Available"].map({"No": 0.0, "Yes": 1.0})
        anxiety_ordinal = frame["Range_Anxiety_Level"].map({"Low": 0.0, "Medium": 1.0, "High": 2.0})
        frame["Home_Charging_Flag"] = home_flag
        frame["Subsidy_Flag"] = subsidy_flag
        frame["Range_Anxiety_Ordinal"] = anxiety_ordinal
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
        frame["Anxiety_x_Commute"] = anxiety_ordinal * frame["Daily_Commute_km"]
        frame["Anxiety_x_Charging"] = anxiety_ordinal * frame["Total_Charging_Stations"]
        frame["Concern_Minus_Anxiety"] = frame["Environmental_Concern_Level"] - anxiety_ordinal
        frame["Subsidy_Range_Context"] = (
            frame["Subsidy_Available"].astype(str) + "_" + frame["Range_Anxiety_Level"].astype(str)
        )
        frame["Home_Range_Context"] = (
            frame["Home_Charging_Possible"].astype(str) + "_" + frame["Range_Anxiety_Level"].astype(str)
        )
        frame["City_Charging_Context"] = (
            frame["City_Type"].astype(str) + "_" + frame["Home_Charging_Possible"].astype(str)
        )
        frame["Concern_Level_Context"] = frame["Environmental_Concern_Level"].fillna(-1).astype(str)

    return frame.replace([np.inf, -np.inf], np.nan)


def make_preprocessor():
    return ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), selector(dtype_include=np.number)),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]),
            selector(dtype_exclude=np.number),
        ),
    ])


def metrics(y_true, probabilities):
    return {
        "ROC_AUC": roc_auc_score(y_true, probabilities),
        "Log_Loss": log_loss(y_true, probabilities),
        "Average_Precision": average_precision_score(y_true, probabilities),
    }


def build_model(parameters, seed, n_estimators=None, early_stopping_rounds=80):
    model_parameters = dict(parameters)
    model_parameters["random_state"] = seed
    if n_estimators is not None:
        model_parameters["n_estimators"] = n_estimators
    if early_stopping_rounds is not None:
        model_parameters["early_stopping_rounds"] = early_stopping_rounds
    return XGBClassifier(**model_parameters)


def save_submission(filename, ids, probabilities, sample_submission):
    submission = pd.DataFrame({ID_COLUMN: ids, TARGET: probabilities})
    assert submission.shape == sample_submission.shape
    assert submission[ID_COLUMN].equals(sample_submission[ID_COLUMN])
    assert submission[TARGET].notna().all()
    assert submission[TARGET].between(0, 1).all()
    path = SUBMISSION_DIR / filename
    submission.to_csv(path, index=False)
    print(f"Saved {path.name}: mean={submission[TARGET].mean():.6f}", flush=True)
    return path


print("Loading competition and original data", flush=True)
competition_train = pd.read_csv(DATA_DIR / "train.csv")
competition_test = pd.read_csv(DATA_DIR / "test.csv")
sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")
original_data = pd.read_csv(ORIGINAL_PATH)

competition_target = competition_train[TARGET].map({"No": 0, "Yes": 1})
competition_features = competition_train.drop(columns=[TARGET])
original_target = original_data[TARGET].map({"No": 0, "Yes": 1})
original_features = original_data.drop(columns=[TARGET])

X_train_tune, X_holdout, y_train_tune, y_holdout = train_test_split(
    competition_features,
    competition_target,
    test_size=0.15,
    random_state=RANDOM_STATE,
    stratify=competition_target,
)
X_train, X_tuning, y_train, y_tuning = train_test_split(
    X_train_tune,
    y_train_tune,
    test_size=0.17647058823529413,
    random_state=RANDOM_STATE + 1,
    stratify=y_train_tune,
)
print(f"Split rows: train={len(X_train)}, tuning={len(X_tuning)}, holdout={len(X_holdout)}", flush=True)

base_parameters = {
    "n_estimators": 1400,
    "learning_rate": 0.025,
    "max_depth": 5,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.0,
    "reg_lambda": 2.0,
    "tree_method": "hist",
    "max_bin": 256,
    "eval_metric": "auc",
    "n_jobs": 4,
}

candidate_parameters = {
    "depth5_reference": base_parameters,
    "depth4_regularized": {
        **base_parameters,
        "n_estimators": 1800,
        "learning_rate": 0.02,
        "max_depth": 4,
        "min_child_weight": 10,
        "reg_alpha": 0.05,
        "reg_lambda": 5.0,
    },
    "depth6_regularized": {
        **base_parameters,
        "n_estimators": 1400,
        "learning_rate": 0.025,
        "max_depth": 6,
        "min_child_weight": 10,
        "gamma": 0.02,
        "reg_alpha": 0.10,
        "reg_lambda": 5.0,
    },
    "lossguide_31": {
        **base_parameters,
        "n_estimators": 1600,
        "learning_rate": 0.02,
        "max_depth": 0,
        "max_leaves": 31,
        "grow_policy": "lossguide",
        "min_child_weight": 10,
        "reg_alpha": 0.05,
        "reg_lambda": 5.0,
    },
}

print("Comparing legacy and refined feature sets", flush=True)
feature_rows = []
feature_artifacts = {}
for feature_name, refined in [("legacy", False), ("refined", True)]:
    train_frame = engineer_features(X_train, refined=refined)
    tuning_frame = engineer_features(X_tuning, refined=refined)
    holdout_frame = engineer_features(X_holdout, refined=refined)
    preprocessor = make_preprocessor()
    train_matrix = preprocessor.fit_transform(train_frame)
    tuning_matrix = preprocessor.transform(tuning_frame)
    holdout_matrix = preprocessor.transform(holdout_frame)
    model = build_model(base_parameters, RANDOM_STATE)
    start = perf_counter()
    model.fit(train_matrix, y_train, eval_set=[(tuning_matrix, y_tuning)], verbose=False)
    tuning_probabilities = model.predict_proba(tuning_matrix)[:, 1]
    holdout_probabilities = model.predict_proba(holdout_matrix)[:, 1]
    row = {
        "Experiment": f"features_{feature_name}",
        "Best_Iteration": model.best_iteration,
        "Fit_Seconds": perf_counter() - start,
        **{f"Tuning_{k}": v for k, v in metrics(y_tuning, tuning_probabilities).items()},
        **{f"Holdout_{k}": v for k, v in metrics(y_holdout, holdout_probabilities).items()},
    }
    feature_rows.append(row)
    feature_artifacts[feature_name] = (preprocessor, train_matrix, tuning_matrix, holdout_matrix)
    print(row, flush=True)

preprocessor, train_matrix, tuning_matrix, holdout_matrix = feature_artifacts["refined"]

print("Searching XGBoost parameter families", flush=True)
parameter_rows = []
parameter_models = {}
for candidate_name, parameters in candidate_parameters.items():
    model = build_model(parameters, RANDOM_STATE)
    start = perf_counter()
    model.fit(train_matrix, y_train, eval_set=[(tuning_matrix, y_tuning)], verbose=False)
    tuning_probabilities = model.predict_proba(tuning_matrix)[:, 1]
    row = {
        "Experiment": f"parameters_{candidate_name}",
        "Candidate": candidate_name,
        "Best_Iteration": model.best_iteration,
        "Fit_Seconds": perf_counter() - start,
        **{f"Tuning_{k}": v for k, v in metrics(y_tuning, tuning_probabilities).items()},
    }
    parameter_rows.append(row)
    parameter_models[candidate_name] = model
    print(row, flush=True)

parameter_results = pd.DataFrame(parameter_rows).sort_values("Tuning_ROC_AUC", ascending=False)
best_candidate_name = parameter_results.iloc[0]["Candidate"]
best_parameters = candidate_parameters[best_candidate_name]
print(f"Best parameter family: {best_candidate_name}", flush=True)

print("Testing original-data augmentation weights", flush=True)
original_frame = engineer_features(original_features, refined=True)
original_matrix = preprocessor.transform(original_frame)
augmentation_rows = []
augmentation_models = {}
for original_weight in [0.0, 1.0, 3.0, 5.0]:
    if original_weight == 0.0:
        augmented_matrix = train_matrix
        augmented_target = y_train.to_numpy()
        sample_weight = None
    else:
        from scipy.sparse import vstack

        augmented_matrix = vstack([train_matrix, original_matrix], format="csr")
        augmented_target = np.concatenate([y_train.to_numpy(), original_target.to_numpy()])
        sample_weight = np.concatenate([
            np.ones(len(y_train), dtype=np.float32),
            np.full(len(original_target), original_weight, dtype=np.float32),
        ])
    model = build_model(best_parameters, RANDOM_STATE)
    start = perf_counter()
    model.fit(
        augmented_matrix,
        augmented_target,
        sample_weight=sample_weight,
        eval_set=[(tuning_matrix, y_tuning)],
        verbose=False,
    )
    tuning_probabilities = model.predict_proba(tuning_matrix)[:, 1]
    holdout_probabilities = model.predict_proba(holdout_matrix)[:, 1]
    row = {
        "Experiment": f"original_weight_{original_weight:g}",
        "Original_Weight": original_weight,
        "Best_Iteration": model.best_iteration,
        "Fit_Seconds": perf_counter() - start,
        **{f"Tuning_{k}": v for k, v in metrics(y_tuning, tuning_probabilities).items()},
        **{f"Holdout_{k}": v for k, v in metrics(y_holdout, holdout_probabilities).items()},
    }
    augmentation_rows.append(row)
    augmentation_models[original_weight] = model
    print(row, flush=True)

augmentation_results = pd.DataFrame(augmentation_rows).sort_values("Tuning_ROC_AUC", ascending=False)
best_original_weight = float(augmentation_results.iloc[0]["Original_Weight"])
best_iteration = int(augmentation_results.iloc[0]["Best_Iteration"]) + 1
print(
    f"Selected original weight={best_original_weight:g}, trees={best_iteration}",
    flush=True,
)

all_experiment_results = pd.concat([
    pd.DataFrame(feature_rows),
    parameter_results,
    augmentation_results,
], ignore_index=True, sort=False)
all_experiment_results.to_csv(ARTIFACT_DIR / "xgboost_optimization_results.csv", index=False)

print("Training three-seed full-data ensemble", flush=True)
full_train_frame = engineer_features(competition_features, refined=True)
test_frame = engineer_features(competition_test, refined=True)
full_original_frame = engineer_features(original_features, refined=True)
final_preprocessor = make_preprocessor()
full_train_matrix = final_preprocessor.fit_transform(full_train_frame)
test_matrix = final_preprocessor.transform(test_frame)
full_original_matrix = final_preprocessor.transform(full_original_frame)

if best_original_weight > 0:
    from scipy.sparse import vstack

    final_train_matrix = vstack([full_train_matrix, full_original_matrix], format="csr")
    final_target = np.concatenate([competition_target.to_numpy(), original_target.to_numpy()])
    final_sample_weight = np.concatenate([
        np.ones(len(competition_target), dtype=np.float32),
        np.full(len(original_target), best_original_weight, dtype=np.float32),
    ])
else:
    final_train_matrix = full_train_matrix
    final_target = competition_target.to_numpy()
    final_sample_weight = None

seed_predictions = []
for seed in [42, 314, 2026]:
    model = build_model(
        best_parameters,
        seed,
        n_estimators=best_iteration,
        early_stopping_rounds=None,
    )
    model.fit(final_train_matrix, final_target, sample_weight=final_sample_weight, verbose=False)
    probabilities = model.predict_proba(test_matrix)[:, 1]
    seed_predictions.append(probabilities)
    save_submission(
        f"submission_xgboost_optimized_seed_{seed}.csv",
        competition_test[ID_COLUMN],
        probabilities,
        sample_submission,
    )

optimized_probabilities = np.mean(seed_predictions, axis=0)
optimized_path = save_submission(
    "submission_xgboost_optimized_seed_ensemble.csv",
    competition_test[ID_COLUMN],
    optimized_probabilities,
    sample_submission,
)

previous_xgboost = pd.read_csv(SUBMISSION_DIR / "submission_xgboost.csv")[TARGET].to_numpy()
previous_catboost = pd.read_csv(SUBMISSION_DIR / "submission_catboost.csv")[TARGET].to_numpy()
previous_lightgbm = pd.read_csv(SUBMISSION_DIR / "submission_lightgbm.csv")[TARGET].to_numpy()

save_submission(
    "submission_optimized_xgb_blend_90_05_05.csv",
    competition_test[ID_COLUMN],
    0.90 * optimized_probabilities + 0.05 * previous_catboost + 0.05 * previous_lightgbm,
    sample_submission,
)
save_submission(
    "submission_optimized_xgb_blend_previous_75_25.csv",
    competition_test[ID_COLUMN],
    0.75 * optimized_probabilities + 0.25 * previous_xgboost,
    sample_submission,
)

metadata = {
    "best_candidate": best_candidate_name,
    "best_original_weight": best_original_weight,
    "best_iteration": best_iteration,
    "seeds": [42, 314, 2026],
    "optimized_submission": str(optimized_path),
}
(ARTIFACT_DIR / "xgboost_optimization_metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)
print("Optimization pipeline completed", flush=True)
