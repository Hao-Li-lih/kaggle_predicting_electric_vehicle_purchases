import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

from pathlib import Path
import random

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TARGET = "Will_Buy_EV"
ID_COLUMN = "id"
BATCH_SIZE = 4096
DEEP_EPOCHS = 7

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.set_num_threads(4)


def engineer_features(data):
    transformed = data.copy().drop(columns=[ID_COLUMN], errors="ignore")
    transformed["Total_Charging_Stations"] = (
        transformed["Charging_Stations_Near_Home"] + transformed["Charging_Stations_Near_Work"]
    )
    transformed["Min_Charging_Stations"] = transformed[
        ["Charging_Stations_Near_Home", "Charging_Stations_Near_Work"]
    ].min(axis=1)
    transformed["Max_Charging_Stations"] = transformed[
        ["Charging_Stations_Near_Home", "Charging_Stations_Near_Work"]
    ].max(axis=1)
    transformed["Home_Work_Station_Gap"] = (
        transformed["Charging_Stations_Near_Home"] - transformed["Charging_Stations_Near_Work"]
    )
    transformed["Commute_Per_Station"] = (
        transformed["Daily_Commute_km"] / (transformed["Total_Charging_Stations"] + 1.0)
    )
    transformed["Log_Annual_Income"] = np.log1p(transformed["Annual_Income_USD"])
    transformed["Income_Per_Car"] = (
        transformed["Annual_Income_USD"] / transformed["Number_of_Cars_Owned"].clip(lower=1)
    )
    transformed["Income_Per_Commute"] = (
        transformed["Annual_Income_USD"] / transformed["Daily_Commute_km"].clip(lower=1)
    )
    transformed["Environment_Income_Interaction"] = (
        transformed["Environmental_Concern_Level"] * transformed["Log_Annual_Income"]
    )
    transformed["Environment_Charging_Interaction"] = (
        transformed["Environmental_Concern_Level"] * transformed["Total_Charging_Stations"]
    )
    transformed["Age_Band"] = pd.cut(
        transformed["Age"],
        bins=[-np.inf, 29, 39, 49, 59, np.inf],
        labels=["25_29", "30_39", "40_49", "50_59", "60_plus"],
    ).astype(str)
    transformed["Income_Band"] = pd.cut(
        transformed["Annual_Income_USD"],
        bins=[-np.inf, 50_000, 75_000, 100_000, 125_000, np.inf],
        labels=["under_50k", "50k_75k", "75k_100k", "100k_125k", "125k_plus"],
    ).astype(str)
    transformed["Commute_Band"] = pd.cut(
        transformed["Daily_Commute_km"],
        bins=[-np.inf, 10, 25, 50, 75, np.inf],
        labels=["under_10", "10_25", "25_50", "50_75", "75_plus"],
    ).astype(str)
    transformed["Charging_Context"] = (
        transformed["Home_Charging_Possible"].astype(str)
        + "_"
        + transformed["Subsidy_Available"].astype(str)
    )
    return transformed.replace([np.inf, -np.inf], np.nan)


class CategoryEmbeddingNetwork(nn.Module):
    def __init__(self, numeric_feature_count, category_cardinalities):
        super().__init__()
        embedding_dimensions = [
            min(16, max(2, int(np.ceil(np.sqrt(cardinality)))))
            for cardinality in category_cardinalities
        ]
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, dimension)
            for cardinality, dimension in zip(category_cardinalities, embedding_dimensions)
        ])
        input_size = numeric_feature_count + sum(embedding_dimensions)
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, numeric_inputs, categorical_inputs):
        embedded_parts = [
            embedding(categorical_inputs[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        combined = torch.cat([numeric_inputs, *embedded_parts], dim=1)
        return self.network(combined).squeeze(1)


def make_loader(numeric_data, categorical_data, labels=None, shuffle=False):
    tensors = [torch.from_numpy(numeric_data), torch.from_numpy(categorical_data)]
    if labels is not None:
        tensors.append(torch.from_numpy(np.asarray(labels, dtype=np.float32)))
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )


def predict_network(model, data_loader):
    model.eval()
    parts = []
    with torch.no_grad():
        for batch in data_loader:
            numeric_batch, categorical_batch = batch[:2]
            parts.append(torch.sigmoid(model(numeric_batch, categorical_batch)).numpy())
    return np.concatenate(parts)


def save_submission(name, ids, probabilities, sample_submission):
    submission = pd.DataFrame({ID_COLUMN: ids, TARGET: probabilities})
    assert submission.shape == sample_submission.shape
    assert submission[ID_COLUMN].equals(sample_submission[ID_COLUMN])
    assert submission[TARGET].notna().all()
    assert submission[TARGET].between(0, 1).all()
    assert submission[TARGET].nunique() > 1
    output_path = SUBMISSION_DIR / name
    submission.to_csv(output_path, index=False)
    print(
        f"Saved {output_path.name}: "
        f"rows={len(submission)}, mean={submission[TARGET].mean():.6f}, "
        f"min={submission[TARGET].min():.6f}, max={submission[TARGET].max():.6f}"
    )


print("Loading data")
train_data = pd.read_csv(DATA_DIR / "train.csv")
test_data = pd.read_csv(DATA_DIR / "test.csv")
sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")
target = train_data[TARGET].map({"No": 0, "Yes": 1})

train_engineered = engineer_features(train_data.drop(columns=[TARGET]))
test_engineered = engineer_features(test_data)
categorical_columns = train_engineered.select_dtypes(exclude=np.number).columns.tolist()
numeric_columns = train_engineered.select_dtypes(include=np.number).columns.tolist()

print("Preparing encoded tree matrices")
tree_preprocessor = ColumnTransformer([
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
train_encoded = tree_preprocessor.fit_transform(train_engineered)
test_encoded = tree_preprocessor.transform(test_engineered)

print("Training XGBoost")
xgboost_model = XGBClassifier(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=5,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=2.0,
    tree_method="hist",
    eval_metric="auc",
    n_jobs=4,
    random_state=RANDOM_STATE,
)
xgboost_model.fit(train_encoded, target, verbose=False)
xgboost_probabilities = xgboost_model.predict_proba(test_encoded)[:, 1]
save_submission(
    "submission_xgboost.csv",
    test_data[ID_COLUMN],
    xgboost_probabilities,
    sample_submission,
)

print("Training LightGBM")
lightgbm_model = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=2.0,
    n_jobs=4,
    random_state=RANDOM_STATE,
    verbosity=-1,
)
lightgbm_model.fit(train_encoded, target)
lightgbm_probabilities = lightgbm_model.predict_proba(test_encoded)[:, 1]
save_submission(
    "submission_lightgbm.csv",
    test_data[ID_COLUMN],
    lightgbm_probabilities,
    sample_submission,
)

print("Training CatBoost")
train_catboost = train_engineered.copy()
test_catboost = test_engineered.copy()
for column in categorical_columns:
    train_catboost[column] = train_catboost[column].fillna("Missing").astype(str)
    test_catboost[column] = test_catboost[column].fillna("Missing").astype(str)

catboost_model = CatBoostClassifier(
    iterations=600,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=5.0,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=RANDOM_STATE,
    thread_count=4,
    verbose=100,
)
catboost_model.fit(train_catboost, target, cat_features=categorical_columns)
catboost_probabilities = catboost_model.predict_proba(test_catboost)[:, 1]
save_submission(
    "submission_catboost.csv",
    test_data[ID_COLUMN],
    catboost_probabilities,
    sample_submission,
)

gbdt_blend_probabilities = np.mean(
    [xgboost_probabilities, catboost_probabilities, lightgbm_probabilities],
    axis=0,
)
save_submission(
    "submission_gbdt_equal_blend.csv",
    test_data[ID_COLUMN],
    gbdt_blend_probabilities,
    sample_submission,
)

print("Preparing neural network tensors")
numeric_imputer = SimpleImputer(strategy="median")
numeric_scaler = StandardScaler()
train_numeric = numeric_scaler.fit_transform(
    numeric_imputer.fit_transform(train_engineered[numeric_columns])
).astype(np.float32)
test_numeric = numeric_scaler.transform(
    numeric_imputer.transform(test_engineered[numeric_columns])
).astype(np.float32)

category_cardinalities = []
train_categorical_parts = []
test_categorical_parts = []
for column in categorical_columns:
    train_values = train_engineered[column].fillna("Missing").astype(str)
    test_values = test_engineered[column].fillna("Missing").astype(str)
    mapping = {value: index + 1 for index, value in enumerate(sorted(train_values.unique()))}
    category_cardinalities.append(len(mapping) + 1)
    train_categorical_parts.append(train_values.map(mapping).fillna(0).to_numpy(dtype=np.int64))
    test_categorical_parts.append(test_values.map(mapping).fillna(0).to_numpy(dtype=np.int64))

train_categorical = np.column_stack(train_categorical_parts)
test_categorical = np.column_stack(test_categorical_parts)
train_loader = make_loader(train_numeric, train_categorical, target, shuffle=True)
test_loader = make_loader(test_numeric, test_categorical, None, shuffle=False)

print("Training PyTorch category embedding network")
network = CategoryEmbeddingNetwork(train_numeric.shape[1], category_cardinalities)
optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3, weight_decay=1e-4)
loss_function = nn.BCEWithLogitsLoss()
for epoch in range(1, DEEP_EPOCHS + 1):
    network.train()
    total_loss = 0.0
    total_examples = 0
    for numeric_batch, categorical_batch, target_batch in train_loader:
        optimizer.zero_grad()
        logits = network(numeric_batch, categorical_batch)
        loss = loss_function(logits, target_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(target_batch)
        total_examples += len(target_batch)
    print(f"Epoch {epoch:02d}: loss={total_loss / total_examples:.6f}")

pytorch_probabilities = predict_network(network, test_loader)
save_submission(
    "submission_pytorch_category_embedding.csv",
    test_data[ID_COLUMN],
    pytorch_probabilities,
    sample_submission,
)

print("All submission files generated successfully")
