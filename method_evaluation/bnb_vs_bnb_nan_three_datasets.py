import io
import os
import re
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")

ALGO_DIR = os.path.join(ROOT_DIR, "algorithms")
if ALGO_DIR not in sys.path:
    sys.path.insert(0, ALGO_DIR)

import bnb
import bnb_nan


K_VALUES = [10]


def _canon(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _resolve_column(df_columns, candidates):
    lookup = {_canon(col): col for col in df_columns}
    for candidate in candidates:
        key = _canon(candidate)
        if key in lookup:
            return lookup[key]
    return None


def _validate_binary_target(y, dataset_name):
    uniq = sorted(pd.unique(y))
    if set(uniq) != {0, 1}:
        raise ValueError(f"{dataset_name}: target must be binary 0/1. Got: {uniq}")


def _cm_rates(cm):
    tn, fp, fn, tp = cm["TN"], cm["FP"], cm["FN"], cm["TP"]
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return sensitivity, specificity


def _format_metric(value, ci):
    return f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def load_breast_cancer():
    path = os.path.join(DATASETS_DIR, "breast_cancer.csv")
    df = pd.read_csv(path)
    raw_rows = len(df)

    if "diagnosis" in df.columns:
        y = pd.to_numeric(df["diagnosis"], errors="coerce")
        X_df = df.drop(columns=["diagnosis"]).copy()
    elif "Class" in df.columns:
        y = pd.to_numeric(df["Class"], errors="coerce").replace({2: 0, 4: 1})
        X_df = df.drop(columns=["Class"]).copy()
        for col in ["id", "ID", "Sample code number"]:
            if col in X_df.columns:
                X_df = X_df.drop(columns=[col])
    else:
        raise ValueError("breast_cancer: expected 'diagnosis' or 'Class' target column.")

    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    valid = (~X_df.isna().any(axis=1)) & (~y.isna())
    X_df = X_df.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(int).reset_index(drop=True)
    _validate_binary_target(y, "breast_cancer")

    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values.astype(float))

    return {
        "name": "Breast Cancer",
        "X": X,
        "y": y.values,
        "method": "BNB",
        "summary": {
            "rows_raw": raw_rows,
            "rows_final": len(X),
            "features": X.shape[1],
            "classes": {"0": int(np.sum(y.values == 0)), "1": int(np.sum(y.values == 1))},
            "input": "scaled clean matrix",
        },
    }


def load_diabetes():
    path = os.path.join(DATASETS_DIR, "diabetes.csv")
    df = pd.read_csv(path)
    raw_rows = len(df)

    target_col = _resolve_column(df.columns, ["Outcome"])
    if target_col is None:
        raise ValueError("diabetes: expected target column 'Outcome'.")

    df_clean = df.drop_duplicates().reset_index(drop=True)
    rows_after_dedup = len(df_clean)

    zero_as_missing_keys = {"glucose", "bloodpressure", "skinthickness", "insulin", "bmi"}
    replaced_cols = []
    for col in df_clean.columns:
        if col == target_col:
            continue
        if _canon(col) in zero_as_missing_keys:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").replace(0, np.nan)
            replaced_cols.append(col)

    y = pd.to_numeric(df_clean[target_col], errors="coerce")
    X_df = df_clean.drop(columns=[target_col]).apply(pd.to_numeric, errors="coerce")

    valid_target = ~y.isna()
    X_df = X_df.loc[valid_target].reset_index(drop=True)
    y = y.loc[valid_target].astype(int).reset_index(drop=True)
    _validate_binary_target(y, "diabetes")

    X_nan = X_df.values.astype(float)

    return {
        "name": "Diabetes",
        "X": X_nan,
        "y": y.values,
        "method": "BNB NaN-aware",
        "summary": {
            "rows_raw": raw_rows,
            "rows_after_dedup": rows_after_dedup,
            "rows_final": len(X_nan),
            "features": X_nan.shape[1],
            "classes": {"0": int(np.sum(y.values == 0)), "1": int(np.sum(y.values == 1))},
            "zero_to_nan_columns": replaced_cols,
            "nan_cells_for_bnb_nan": int(np.isnan(X_nan).sum()),
            "input": "NaN-preserving matrix (zero-as-missing)",
        },
    }


def load_heart_disease():
    path = os.path.join(DATASETS_DIR, "heart_disease.csv")
    df = pd.read_csv(path)
    raw_rows = len(df)

    df = df.drop_duplicates().reset_index(drop=True)
    rows_after_dedup = len(df)

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

    encoded_cols = []
    for col in X_df.columns:
        if not pd.api.types.is_numeric_dtype(X_df[col]):
            X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))
            encoded_cols.append(col)

    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    valid = (~X_df.isna().any(axis=1)) & (~y.isna())
    X_df = X_df.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(int).reset_index(drop=True)
    _validate_binary_target(y, "heart_disease")

    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values.astype(float))

    return {
        "name": "Heart Disease",
        "X": X,
        "y": y.values,
        "method": "BNB",
        "summary": {
            "rows_raw": raw_rows,
            "rows_after_dedup": rows_after_dedup,
            "rows_final": len(X),
            "features": X.shape[1],
            "classes": {"0": int(np.sum(y.values == 0)), "1": int(np.sum(y.values == 1))},
            "encoded_columns": encoded_cols,
            "input": "scaled clean matrix",
        },
    }


def evaluate_dataset(dataset):
    name = dataset["name"]
    X = dataset["X"]
    y = dataset["y"]
    method = dataset["method"]
    summary = dataset["summary"]

    if method == "BNB":
        with redirect_stdout(io.StringIO()):
            res = bnb.evaluate(X, y, k_values=K_VALUES)
    elif method == "BNB NaN-aware":
        with redirect_stdout(io.StringIO()):
            res = bnb_nan.evaluate(X, y, k_values=K_VALUES)
    else:
        raise ValueError(f"Unsupported method: {method}")

    k = 10
    r = res[k]
    sens, spec = _cm_rates(r["confusion_matrix"])

    print(f"\n{'=' * 112}")
    print(f"{name} | k=10")
    print(f"{'=' * 112}")
    print(
        f"Rows: {summary['rows_final']} | Features: {summary['features']} | "
        f"Class(0/1): {summary['classes']['0']}/{summary['classes']['1']}"
    )
    if name == "Diabetes":
        print(f"Rows raw: {summary['rows_raw']} | After dedup: {summary['rows_after_dedup']}")
        print(f"Zero-as-missing columns: {summary['zero_to_nan_columns']}")
        print(f"NaN cells used by BNB NaN-aware: {summary['nan_cells_for_bnb_nan']}")
    elif name == "Heart Disease":
        print(f"Rows raw: {summary['rows_raw']} | After dedup: {summary['rows_after_dedup']}")
    else:
        print(f"Rows raw: {summary['rows_raw']}")

    print(f"Method: {method}")
    print(f"Input: {summary['input']}")
    print(f"{'-' * 112}")
    print(f"{'Method':<22} {'Accuracy':<24} {'Precision':<24} {'Recall':<24} {'F1':<24} {'AUC':<24}")
    print(f"{'-' * 140}")
    print(
        f"{method:<22} "
        f"{_format_metric(r['accuracy'], r['ci']['accuracy']):<24} "
        f"{_format_metric(r['precision'], r['ci']['precision']):<24} "
        f"{_format_metric(r['recall'], r['ci']['recall']):<24} "
        f"{_format_metric(r['f1'], r['ci']['f1']):<24} "
        f"{_format_metric(r['auc'], r['ci']['auc']):<24}"
    )
    print(f"{'-' * 112}")
    print(
        f"Confusion matrix: TN={r['confusion_matrix']['TN']} FP={r['confusion_matrix']['FP']} "
        f"FN={r['confusion_matrix']['FN']} TP={r['confusion_matrix']['TP']} "
        f"| Sens={sens:.3f} Spec={spec:.3f}"
    )


def main():
    print("BNB evaluation across three datasets")
    print("Diabetes uses BNB NaN-aware")
    print("Cross-validation folds: k=10")

    datasets = [
        load_breast_cancer(),
        load_diabetes(),
        load_heart_disease(),
    ]

    for dataset in datasets:
        evaluate_dataset(dataset)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
