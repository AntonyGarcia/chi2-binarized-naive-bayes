"""
Stability of chi2-selected binarization thresholds under repeated stratified
10-fold cross-validation (reviewer comments R1.4, R2.2, R3.5).

For every training partition of a 10 x 10 repeated stratified CV, the chi2
threshold of each continuous feature is re-estimated exactly as in the main
evaluation pipeline (scipy chi2_contingency, Yates-corrected, x > t split).
For each feature we report, in original clinical units:
  - the threshold selected on the full dataset,
  - the median and 2.5-97.5 percentile range of the 100 fold-level thresholds,
  - the percentage of folds that select the full-data threshold,
  - the mean percentage of patients whose binary state is unchanged when the
    fold threshold is used instead of the full-data threshold.
The pooled out-of-fold AUC of chi2-BNB is also reported for each repeat.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
ALGO_DIR = os.path.join(ROOT_DIR, "algorithms")
for path in (BASE_DIR, ALGO_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import bnb_binarization_variants_three_datasets as variants
import chi2_feature_summary_tables as summary
from bnb import encode_categorical


SEED = 42
K = 10
N_REPEATS = 10
OUT_DIR = os.path.join(BASE_DIR, "outputs")

HEART_CONTINUOUS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]


def chi2_yates_threshold(x_col, y):
    """Vectorised equivalent of variants._best_threshold_by_score with _chi2_score."""
    valid = ~np.isnan(x_col)
    x = x_col[valid]
    yv = y[valid].astype(int)
    if len(x) == 0:
        return 0.0

    uniq, inv = np.unique(x, return_inverse=True)
    pos_per = np.bincount(inv, weights=(yv == 1), minlength=len(uniq))
    neg_per = np.bincount(inv, weights=(yv == 0), minlength=len(uniq))

    # Row 0: x <= t, row 1: x > t; columns: y = 0, y = 1.
    a = np.cumsum(neg_per)
    b = np.cumsum(pos_per)
    c = a[-1] - a
    d = b[-1] - b
    n = float(len(x))

    denom = (a + b) * (c + d) * (a + c) * (b + d)
    score = np.full(len(uniq), -np.inf)
    ok = denom > 0
    # scipy's Yates correction shrinks |ad - bc| by n/2 but never below zero.
    diff = np.maximum(np.abs(a * d - b * c) - n / 2.0, 0.0)
    score[ok] = n * diff[ok] ** 2 / denom[ok]

    if not np.any(np.isfinite(score)):
        return float(uniq[0])
    return float(uniq[int(np.argmax(score))])


def load_datasets():
    """Pair each CV-pipeline dataset with its raw-unit feature matrix."""
    pairs = [
        (variants.load_breast_cancer(), summary.load_breast_cancer(), None),
        (variants.load_diabetes(), summary.load_diabetes(), None),
        (variants.load_heart_disease(), summary.load_heart_disease(), HEART_CONTINUOUS),
    ]
    datasets = []
    for cv_data, (_, raw_df, raw_y), continuous in pairs:
        raw = raw_df.to_numpy(dtype=float)
        if not np.array_equal(cv_data["y"], raw_y):
            raise RuntimeError(f"{cv_data['name']}: row order differs between loaders.")
        # Breast Cancer and Heart Disease are standardised in the CV pipeline;
        # thresholds are monotone-equivalent, so we work in raw units.
        if not cv_data["use_nan_aware"]:
            if not np.allclose(StandardScaler().fit_transform(raw), cv_data["X"]):
                raise RuntimeError(f"{cv_data['name']}: raw and CV feature matrices differ.")
        names = list(raw_df.columns)
        report_idx = [names.index(f) for f in continuous] if continuous else list(range(len(names)))
        datasets.append(
            {
                "name": cv_data["name"],
                "X": raw,
                "y": raw_y,
                "use_nan_aware": cv_data["use_nan_aware"],
                "categorical_idx": cv_data.get("categorical_idx"),
                "features": names,
                "report_idx": report_idx,
            }
        )
    return datasets


def run_dataset(ds):
    X, y = ds["X"], ds["y"]
    n_features = X.shape[1]
    full_thr = np.array([chi2_yates_threshold(X[:, j], y) for j in range(n_features)])

    # Categorical columns (Heart Disease) are modelled with Categorical NB.
    categorical_idx = ds["categorical_idx"]
    X_model, n_categories = encode_categorical(X, categorical_idx) if categorical_idx else (X, None)

    fold_thr = []
    repeat_auc = []
    for r in range(N_REPEATS):
        cv = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED + r)
        y_true, y_prob = [], []
        for train_idx, test_idx in cv.split(X, y):
            thr = np.array([chi2_yates_threshold(X[train_idx, j], y[train_idx]) for j in range(n_features)])
            fold_thr.append(thr)

            model, X_test_bin = variants.fit_binarized_model(
                X_model[train_idx], y[train_idx], X_model[test_idx], variants.SPLIT_METHODS["chi2"],
                ds["use_nan_aware"], categorical_idx, n_categories,
            )
            pos = list(model.classes_).index(1)
            y_true.append(y[test_idx])
            y_prob.append(model.predict_proba(X_test_bin)[:, pos])
        repeat_auc.append(variants._safe_auc(np.concatenate(y_true), np.concatenate(y_prob)))

    fold_thr = np.array(fold_thr)
    rows = []
    for j in ds["report_idx"]:
        x = X[:, j]
        valid = ~np.isnan(x)
        side_full = x[valid] > full_thr[j]
        agreement = [np.mean((x[valid] > t) == side_full) for t in fold_thr[:, j]]
        rows.append(
            {
                "Dataset": ds["name"],
                "Feature": ds["features"][j],
                "Full-data threshold": full_thr[j],
                "Fold median": float(np.median(fold_thr[:, j])),
                "Fold P2.5": float(np.percentile(fold_thr[:, j], 2.5)),
                "Fold P97.5": float(np.percentile(fold_thr[:, j], 97.5)),
                "% folds = full-data threshold": 100.0 * float(np.mean(np.isclose(fold_thr[:, j], full_thr[j]))),
                "% patients same side (mean)": 100.0 * float(np.mean(agreement)),
                "% patients same side (min)": 100.0 * float(np.min(agreement)),
            }
        )
    return pd.DataFrame(rows), np.array(repeat_auc)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tables, auc_rows = [], []
    for ds in load_datasets():
        table, aucs = run_dataset(ds)
        tables.append(table)
        auc_rows.append(
            {
                "Dataset": ds["name"],
                "Mean AUC": aucs.mean(),
                "SD AUC": aucs.std(ddof=1),
                "Min AUC": aucs.min(),
                "Max AUC": aucs.max(),
            }
        )
        print(f"\n{'=' * 120}\n{ds['name']} | {N_REPEATS} x {K}-fold stratified CV ({N_REPEATS * K} training sets)\n{'=' * 120}")
        print(table.drop(columns="Dataset").to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"AUC per repeat: {np.round(aucs, 3)}")

    thr_table = pd.concat(tables, ignore_index=True)
    auc_table = pd.DataFrame(auc_rows)
    thr_table.to_csv(os.path.join(OUT_DIR, "threshold_stability.csv"), index=False)
    auc_table.to_csv(os.path.join(OUT_DIR, "threshold_stability_auc.csv"), index=False)
    print(f"\n{auc_table.to_string(index=False, float_format=lambda v: f'{v:.3f}')}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
