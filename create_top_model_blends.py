from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SUBMISSION_DIR = ROOT / "submissions"
TARGET = "Will_Buy_EV"
ID_COLUMN = "id"


def percentile_rank(values):
    return values.rank(method="average", pct=True)


def save_submission(filename, ids, predictions, sample_submission):
    submission = pd.DataFrame({ID_COLUMN: ids, TARGET: predictions})
    assert submission.shape == sample_submission.shape
    assert submission[ID_COLUMN].equals(sample_submission[ID_COLUMN])
    assert submission[TARGET].notna().all()
    assert submission[TARGET].between(0, 1).all()
    assert submission[TARGET].nunique() > 1
    path = SUBMISSION_DIR / filename
    submission.to_csv(path, index=False)
    print(f"Saved {path.name}: mean={submission[TARGET].mean():.6f}")


sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")
finebin_4096 = pd.read_csv(
    SUBMISSION_DIR / "submission_xgboost_finebin_4096_seed_ensemble.csv"
)
finebin_8192 = pd.read_csv(
    SUBMISSION_DIR / "submission_xgboost_finebin_8192_seed_42.csv"
)

assert finebin_4096[ID_COLUMN].equals(finebin_8192[ID_COLUMN])

probability_75_25 = (
    0.75 * finebin_4096[TARGET]
    + 0.25 * finebin_8192[TARGET]
)
probability_90_10 = (
    0.90 * finebin_4096[TARGET]
    + 0.10 * finebin_8192[TARGET]
)
rank_75_25 = (
    0.75 * percentile_rank(finebin_4096[TARGET])
    + 0.25 * percentile_rank(finebin_8192[TARGET])
)

save_submission(
    "submission_top2_probability_blend_75_25.csv",
    sample_submission[ID_COLUMN],
    probability_75_25,
    sample_submission,
)
save_submission(
    "submission_top2_probability_blend_90_10.csv",
    sample_submission[ID_COLUMN],
    probability_90_10,
    sample_submission,
)
save_submission(
    "submission_top2_rank_blend_75_25.csv",
    sample_submission[ID_COLUMN],
    rank_75_25,
    sample_submission,
)
