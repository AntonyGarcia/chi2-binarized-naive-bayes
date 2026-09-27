import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
ALGO_DIR = os.path.join(ROOT_DIR, "algorithms")

if ALGO_DIR not in sys.path:
    sys.path.insert(0, ALGO_DIR)

import bnb


SEED = 42
N_BOOT = 1000
MIN_OOB = 20


def _canon(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _resolve_column(df_columns, candidates):
    lookup = {_canon(c): c for c in df_columns}
    for candidate in candidates:
        key = _canon(candidate)
        if key in lookup:
            return lookup[key]
    return None


def _validate_binary_target(y, dataset_name):
    uniq = sorted(pd.unique(y))
    if set(uniq) != {0, 1}:
        raise ValueError(f"{dataset_name}: target must be binary 0/1. Got: {uniq}")


def load_breast_cancer():
    path = os.path.join(DATASETS_DIR, "breast_cancer.csv")
    df = pd.read_csv(path)

    if "diagnosis" in df.columns:
        y = df["diagnosis"].copy()
        if y.dtype == object:
            y = y.map({"M": 1, "B": 0, "malignant": 1, "benign": 0})
        X_df = df.drop(columns=["diagnosis"]).copy()
    elif "Class" in df.columns:
        y = pd.to_numeric(df["Class"], errors="coerce").replace({2: 0, 4: 1})
        X_df = df.drop(columns=["Class"]).copy()
        for col in ["id", "ID", "Sample code number"]:
            if col in X_df.columns:
                X_df = X_df.drop(columns=[col])
    else:
        raise ValueError("breast_cancer: expected target column 'diagnosis' or 'Class'.")

    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    valid = (~X_df.isna().any(axis=1)) & (~y.isna())

    X = X_df.loc[valid].to_numpy(dtype=float)
    y = y.loc[valid].astype(int).to_numpy()
    _validate_binary_target(y, "breast_cancer")
    return "breast_cancer", X, y


def load_diabetes():
    path = os.path.join(DATASETS_DIR, "diabetes.csv")
    df = pd.read_csv(path).drop_duplicates().reset_index(drop=True)

    target_col = _resolve_column(df.columns, ["Outcome"])
    if target_col is None:
        raise ValueError("diabetes: expected target column 'Outcome'.")

    zero_as_missing_keys = {"glucose", "bloodpressure", "skinthickness", "insulin", "bmi"}
    for col in df.columns:
        if col == target_col:
            continue
        if _canon(col) in zero_as_missing_keys:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace(0, np.nan)

    y = pd.to_numeric(df[target_col], errors="coerce")
    X_df = df.drop(columns=[target_col]).apply(pd.to_numeric, errors="coerce")
    valid_target = ~y.isna()

    X = X_df.loc[valid_target].to_numpy(dtype=float)
    y = y.loc[valid_target].astype(int).to_numpy()
    _validate_binary_target(y, "diabetes")
    return "diabetes", X, y


def load_heart_disease():
    path = os.path.join(DATASETS_DIR, "heart_disease.csv")
    df = pd.read_csv(path).drop_duplicates().reset_index(drop=True)

    target_col = _resolve_column(df.columns, ["HeartDisease"])
    if target_col is None:
        raise ValueError("heart_disease: expected target column 'HeartDisease'.")

    rbp_col = _resolve_column(df.columns, ["RestingBP"])
    if rbp_col is not None:
        df[rbp_col] = pd.to_numeric(df[rbp_col], errors="coerce")
        df = df[df[rbp_col] >= 80].copy()

    chol_col = _resolve_column(df.columns, ["Cholesterol"])
    if chol_col is not None:
        df[chol_col] = pd.to_numeric(df[chol_col], errors="coerce")
        nonzero = df.loc[df[chol_col] > 0, chol_col]
        if len(nonzero) > 0:
            df.loc[df[chol_col] == 0, chol_col] = nonzero.median()
        df = df[(df[chol_col] >= 100) & (df[chol_col] <= 450)].copy()

    y = pd.to_numeric(df[target_col], errors="coerce")
    X_df = df.drop(columns=[target_col]).copy()
    for col in X_df.columns:
        if not pd.api.types.is_numeric_dtype(X_df[col]):
            X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))
    X_df = X_df.apply(pd.to_numeric, errors="coerce")

    valid = (~X_df.isna().any(axis=1)) & (~y.isna())
    X = X_df.loc[valid].to_numpy(dtype=float)
    y = y.loc[valid].astype(int).to_numpy()
    _validate_binary_target(y, "heart_disease")
    return "heart_disease", X, y


def _bootstrap_oob_bnb(X, y, n_boot=N_BOOT, seed=SEED, min_oob=MIN_OOB):
    rng = np.random.default_rng(seed)
    n = len(y)

    metric_store = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "auc": [],
    }
    used_iters = 0
    skipped_iters = 0

    for _ in range(n_boot):
        train_idx = rng.integers(0, n, size=n)
        in_bag = np.zeros(n, dtype=bool)
        in_bag[train_idx] = True
        test_idx = np.flatnonzero(~in_bag)

        if len(test_idx) < min_oob:
            skipped_iters += 1
            continue

        y_train = y[train_idx]
        y_test = y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            skipped_iters += 1
            continue

        X_train = X[train_idx]
        X_test = X[test_idx]

        # BNB implementation is not NaN-aware; impute per bootstrap replicate.
        if np.isnan(X_train).any() or np.isnan(X_test).any():
            imputer = SimpleImputer(strategy="mean")
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)

        scores = bnb.train_and_score(X_train, y_train, X_test, y_test)
        metric_store["accuracy"].append(float(scores["accuracy"]))
        metric_store["precision"].append(float(scores["precision"]))
        metric_store["recall"].append(float(scores["recall"]))
        metric_store["f1"].append(float(scores["f1"]))
        metric_store["auc"].append(float(scores["auc"]))
        used_iters += 1

    if used_iters == 0:
        raise RuntimeError("No valid bootstrap iterations were produced. Try lowering --min_oob.")

    def summarize(values):
        arr = np.asarray(values, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "ci": (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))),
        }

    summary = {m: summarize(v) for m, v in metric_store.items()}
    return summary, used_iters, skipped_iters


def _print_summary(dataset_name, summary, used_iters, skipped_iters, requested_boot):
    print(f"\n{'=' * 108}")
    print(f"{dataset_name} | Bootstrap OOB performance | BNB")
    print(f"{'=' * 108}")
    print(f"Requested bootstraps: {requested_boot} | Used: {used_iters} | Skipped: {skipped_iters}")
    print(f"{'-' * 108}")
    print(f"{'Metric':<12} {'Mean':<12} {'Std':<12} {'95% CI'}")
    print(f"{'-' * 108}")

    for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
        s = summary[metric]
        lo, hi = s["ci"]
        print(f"{metric:<12} {s['mean']:<12.4f} {s['std']:<12.4f} [{lo:.4f}, {hi:.4f}]")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap out-of-bag performance for BNB. Each bootstrap draws n samples with replacement "
            "for training, then evaluates on OOB samples; reports mean/std and 95% percentile CI."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["breast_cancer", "diabetes", "heart_disease", "all"],
        default="all",
        help="Dataset to evaluate (default: all).",
    )
    parser.add_argument("--n_boot", type=int, default=N_BOOT, help="Number of bootstrap iterations (default: 1000).")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42).")
    parser.add_argument("--min_oob", type=int, default=MIN_OOB, help="Minimum OOB samples required per iteration.")
    args = parser.parse_args()

    loader_map = {
        "breast_cancer": load_breast_cancer,
        "diabetes": load_diabetes,
        "heart_disease": load_heart_disease,
    }
    run_order = list(loader_map.keys()) if args.dataset == "all" else [args.dataset]

    print("BNB bootstrap performance (out-of-bag)")
    for key in run_order:
        dataset_name, X, y = loader_map[key]()
        summary, used_iters, skipped_iters = _bootstrap_oob_bnb(
            X, y, n_boot=args.n_boot, seed=args.seed, min_oob=args.min_oob
        )
        _print_summary(dataset_name, summary, used_iters, skipped_iters, args.n_boot)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
