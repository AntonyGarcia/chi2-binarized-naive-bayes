import os
import re
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mutual_info_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB, CategoricalNB
from sklearn.preprocessing import LabelEncoder, StandardScaler


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
ALGO_DIR = os.path.join(ROOT_DIR, "algorithms")

if ALGO_DIR not in sys.path:
    sys.path.insert(0, ALGO_DIR)

import bnb_nan
from bnb import bootstrap_ci, categorical_indices, delong_test, encode_categorical, mcnemar_test


SEED = 42
K = 10
BOOTSTRAP_N = 1000


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


def _safe_auc(y_true, y_score):
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _entropy_binary(y):
    if len(y) == 0:
        return 0.0
    p1 = float(np.mean(y == 1))
    p0 = 1.0 - p1
    eps = 1e-12
    e = 0.0
    if p0 > 0:
        e -= p0 * np.log2(p0 + eps)
    if p1 > 0:
        e -= p1 * np.log2(p1 + eps)
    return float(e)


def _gini_binary(y):
    if len(y) == 0:
        return 0.0
    p1 = float(np.mean(y == 1))
    p0 = 1.0 - p1
    return float(1.0 - p0 * p0 - p1 * p1)


def _chi2_score(binary_feature, y):
    table = np.zeros((2, 2), dtype=int)
    for b, label in zip(binary_feature, y):
        table[int(b), int(label)] += 1

    # Degenerate split or labels.
    if table[0].sum() == 0 or table[1].sum() == 0:
        return float("-inf")
    if table[:, 0].sum() == 0 or table[:, 1].sum() == 0:
        return float("-inf")

    chi2_stat, _, _, _ = chi2_contingency(table)
    return float(chi2_stat)


def _mutual_information_score(binary_feature, y):
    if len(np.unique(binary_feature)) < 2 or len(np.unique(y)) < 2:
        return float("-inf")
    return float(mutual_info_score(y, binary_feature))


def _gini_gain_score(binary_feature, y):
    if len(np.unique(binary_feature)) < 2:
        return float("-inf")

    parent = _gini_binary(y)
    left = y[binary_feature == 0]
    right = y[binary_feature == 1]
    if len(left) == 0 or len(right) == 0:
        return float("-inf")

    weighted = (len(left) / len(y)) * _gini_binary(left) + (len(right) / len(y)) * _gini_binary(right)
    return float(parent - weighted)


def _information_gain_score(binary_feature, y):
    if len(np.unique(binary_feature)) < 2:
        return float("-inf")

    parent = _entropy_binary(y)
    left = y[binary_feature == 0]
    right = y[binary_feature == 1]
    if len(left) == 0 or len(right) == 0:
        return float("-inf")

    weighted = (len(left) / len(y)) * _entropy_binary(left) + (len(right) / len(y)) * _entropy_binary(right)
    return float(parent - weighted)


def _auc_split_score(binary_feature, y):
    if len(np.unique(y)) < 2:
        return float("-inf")
    try:
        auc = roc_auc_score(y, binary_feature)
        return float(max(auc, 1.0 - auc))
    except ValueError:
        return float("-inf")


def _youden_j_score(binary_feature, y):
    if len(np.unique(binary_feature)) < 2:
        return float("-inf")

    tp = int(np.sum((binary_feature == 1) & (y == 1)))
    tn = int(np.sum((binary_feature == 0) & (y == 0)))
    fp = int(np.sum((binary_feature == 1) & (y == 0)))
    fn = int(np.sum((binary_feature == 0) & (y == 1)))

    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return float(tpr + tnr - 1.0)


def _mean_threshold(x_col, _y_unused):
    valid = x_col[~np.isnan(x_col)]
    if len(valid) == 0:
        return 0.0
    return float(np.mean(valid))


def _median_threshold(x_col, _y_unused):
    valid = x_col[~np.isnan(x_col)]
    if len(valid) == 0:
        return 0.0
    return float(np.median(valid))


def _otsu_threshold(x_col, _y_unused):
    valid = x_col[~np.isnan(x_col)]
    if len(valid) == 0:
        return 0.0
    if len(np.unique(valid)) == 1:
        return float(valid[0])

    bins = min(256, max(16, int(np.sqrt(len(valid)))))
    counts, edges = np.histogram(valid, bins=bins)
    mids = (edges[:-1] + edges[1:]) / 2.0

    w1 = np.cumsum(counts).astype(float)
    w2 = np.cumsum(counts[::-1])[::-1].astype(float)

    sum_total = np.sum(counts * mids)
    sum_b = np.cumsum(counts * mids)

    m1 = np.divide(sum_b, w1, out=np.zeros_like(sum_b), where=w1 > 0)
    m2 = np.divide(sum_total - sum_b, w2, out=np.zeros_like(sum_b), where=w2 > 0)

    between = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    if len(between) == 0:
        return float(np.median(valid))

    idx = int(np.argmax(between))
    return float(mids[idx])


def _kmeans_midpoint_threshold(x_col, _y_unused):
    valid = x_col[~np.isnan(x_col)]
    if len(valid) == 0:
        return 0.0
    uniq = np.unique(valid)
    if len(uniq) == 1:
        return float(uniq[0])

    km = KMeans(n_clusters=2, random_state=SEED, n_init=10)
    km.fit(valid.reshape(-1, 1))
    centers = np.sort(km.cluster_centers_.ravel())
    return float((centers[0] + centers[1]) / 2.0)


SPLIT_METHODS = OrderedDict(
    [
        ("chi2", {"kind": "score", "fn": _chi2_score}),
        ("mutual_info", {"kind": "score", "fn": _mutual_information_score}),
        ("gini_gain", {"kind": "score", "fn": _gini_gain_score}),
        ("information_gain", {"kind": "score", "fn": _information_gain_score}),
        ("auc_split", {"kind": "score", "fn": _auc_split_score}),
        ("otsu", {"kind": "direct", "fn": _otsu_threshold}),
        ("median", {"kind": "direct", "fn": _median_threshold}),
        ("mean", {"kind": "direct", "fn": _mean_threshold}),
        ("youden_j", {"kind": "score", "fn": _youden_j_score}),
        ("kmeans_midpoint", {"kind": "direct", "fn": _kmeans_midpoint_threshold}),
    ]
)


def _best_threshold_by_score(x_col, y, score_fn):
    valid = ~np.isnan(x_col)
    x_valid = x_col[valid]
    y_valid = y[valid]

    if len(x_valid) == 0:
        return 0.0

    candidates = np.unique(x_valid)
    best_t = float(candidates[0])
    best_score = float("-inf")

    for t in candidates:
        binary = (x_valid > t).astype(int)
        score = score_fn(binary, y_valid)
        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t


def _compute_thresholds(X_train, y_train, method):
    thresholds = []
    kind = method["kind"]
    fn = method["fn"]

    for j in range(X_train.shape[1]):
        x_col = X_train[:, j]
        if kind == "score":
            t = _best_threshold_by_score(x_col, y_train, fn)
        elif kind == "direct":
            t = fn(x_col, y_train)
        else:
            raise ValueError(f"Unknown threshold method kind: {kind}")
        thresholds.append(t)

    return np.array(thresholds, dtype=float)


def _binarize(X, thresholds, keep_nan):
    if keep_nan:
        return np.where(np.isnan(X), np.nan, (X > thresholds).astype(float))
    return (X > thresholds).astype(float)


def fit_binarized_model(X_train, y_train, X_test, method, use_nan_aware, categorical_idx=None, n_categories=None):
    """
    Estimate thresholds on the training fold, binarize, and fit the Naive Bayes model.

    With categorical_idx, only the other columns are thresholded and the model is
    Categorical NB (categorical columns must already be integer-coded, see
    bnb.encode_categorical). Returns the fitted model and the encoded test matrix.
    """
    if categorical_idx:
        cont = np.ones(X_train.shape[1], dtype=bool)
        cont[list(categorical_idx)] = False
        thresholds = _compute_thresholds(X_train[:, cont], y_train, method)
        X_train_enc = X_train.copy()
        X_test_enc = X_test.copy()
        X_train_enc[:, cont] = X_train[:, cont] > thresholds
        X_test_enc[:, cont] = X_test[:, cont] > thresholds
        model = CategoricalNB(alpha=1.0, min_categories=n_categories)
        model.fit(X_train_enc.astype(int), y_train)
        return model, X_test_enc.astype(int)

    thresholds = _compute_thresholds(X_train, y_train, method)
    X_train_bin = _binarize(X_train, thresholds, keep_nan=use_nan_aware)
    X_test_bin = _binarize(X_test, thresholds, keep_nan=use_nan_aware)
    model = bnb_nan.BernoulliNB_NaN() if use_nan_aware else BernoulliNB(binarize=None)
    model.fit(X_train_bin, y_train)
    return model, X_test_bin


def evaluate_variant(X, y, method, use_nan_aware, categorical_idx=None, seed=SEED):
    cv = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    n_categories = None
    if categorical_idx:
        X, n_categories = encode_categorical(X, categorical_idx)

    for train_idx, test_idx in cv.split(X, y):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        model, X_test_bin = fit_binarized_model(
            X_train, y_train, X_test, method, use_nan_aware, categorical_idx, n_categories
        )
        y_pred = model.predict(X_test_bin)
        proba = model.predict_proba(X_test_bin)
        classes = list(model.classes_)
        pos_idx = classes.index(1)
        y_prob = proba[:, pos_idx]

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        y_prob_all.append(y_prob)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    y_prob_all = np.concatenate(y_prob_all)

    tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ci = bootstrap_ci(y_true_all, y_pred_all, y_prob_all, n_boot=BOOTSTRAP_N, ci=95, random_state=SEED)

    return {
        "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
        "precision": float(precision_score(y_true_all, y_pred_all, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true_all, y_pred_all, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0)),
        "auc": _safe_auc(y_true_all, y_prob_all),
        "ci": ci,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "cm": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "y_prob": y_prob_all,
    }


def _fmt_metric_ci(value, ci_tuple):
    return f"{value:.3f} [{ci_tuple[0]:.3f}, {ci_tuple[1]:.3f}]"


def _print_pairwise_tests_vs_bnb(results, baseline_key="chi2"):
    if baseline_key not in results:
        print("\nPairwise tests skipped: baseline chi2 result missing.")
        return

    baseline = results[baseline_key]

    print(f"\n{'=' * 110}")
    print("Pairwise DeLong Test vs BNB baseline (chi2)")
    print(f"{'=' * 110}")
    print(f"{'Variant':<20} {'AUC_variant':<12} {'AUC_bnb':<10} {'Diff':<10} {'Z':<10} {'p-value':<12} {'Significant?'}")
    print(f"{'-' * 110}")

    for variant_name in SPLIT_METHODS.keys():
        if variant_name == baseline_key:
            continue
        current = results[variant_name]
        stat = delong_test(
            baseline["y_true"],
            current["y_prob"],
            baseline["y_prob"],
        )
        sig = "Yes" if stat["p_value"] < 0.05 else "No"
        print(
            f"{variant_name:<20} {stat['auc_a']:<12.3f} {stat['auc_b']:<10.3f} "
            f"{stat['diff']:+.3f}    {stat['z_stat']:<10.3f} {stat['p_value']:<12.4f} {sig}"
        )

    print(f"\n{'=' * 110}")
    print("Pairwise McNemar Test vs BNB baseline (chi2)")
    print(f"{'=' * 110}")
    print(
        f"{'Variant':<20} {'b (var+ bnb-)':<16} {'c (var- bnb+)':<16} "
        f"{'Statistic':<12} {'p-value':<12} {'Method':<8} {'Significant?'}"
    )
    print(f"{'-' * 110}")

    for variant_name in SPLIT_METHODS.keys():
        if variant_name == baseline_key:
            continue
        current = results[variant_name]
        stat = mcnemar_test(
            baseline["y_true"],
            current["y_pred"],
            baseline["y_pred"],
        )
        sig = "Yes" if stat["p_value"] < 0.05 else "No"
        stat_str = f"{stat['statistic']:.3f}" if not np.isnan(stat["statistic"]) else "-"
        print(
            f"{variant_name:<20} {stat['b']:<16} {stat['c']:<16} "
            f"{stat_str:<12} {stat['p_value']:<12.4f} {stat['method']:<8} {sig}"
        )


def load_breast_cancer():
    path = os.path.join(DATASETS_DIR, "breast_cancer.csv")
    df = pd.read_csv(path)
    raw_rows = len(df)

    if "diagnosis" in df.columns:
        y = df["diagnosis"].copy()
        X_df = df.drop(columns=["diagnosis"]).copy()
        if y.dtype == object:
            y = y.map({"M": 1, "B": 0, "malignant": 1, "benign": 0})
    elif "Class" in df.columns:
        y = pd.to_numeric(df["Class"], errors="coerce").replace({2: 0, 4: 1})
        X_df = df.drop(columns=["Class"]).copy()
        for id_col in ["id", "ID", "Sample code number"]:
            if id_col in X_df.columns:
                X_df = X_df.drop(columns=[id_col])
    else:
        raise ValueError("breast_cancer: expected 'diagnosis' or 'Class' target column.")

    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    valid = (~X_df.isna().any(axis=1)) & (~y.isna())
    X_df = X_df.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(int).reset_index(drop=True)
    _validate_binary_target(y, "breast_cancer")

    X = StandardScaler().fit_transform(X_df.values.astype(float))
    return {
        "name": "Breast Cancer",
        "X": X,
        "y": y.values,
        "use_nan_aware": False,
        "rows_raw": raw_rows,
        "rows_final": len(X),
        "features": X.shape[1],
    }


def load_diabetes():
    path = os.path.join(DATASETS_DIR, "diabetes.csv")
    df = pd.read_csv(path)
    raw_rows = len(df)

    target_col = _resolve_column(df.columns, ["Outcome"])
    if target_col is None:
        raise ValueError("diabetes: expected target column 'Outcome'.")

    df = df.drop_duplicates().reset_index(drop=True)
    rows_after_dedup = len(df)

    zero_as_missing_keys = {"glucose", "bloodpressure", "skinthickness", "insulin", "bmi"}
    for col in df.columns:
        if col == target_col:
            continue
        if _canon(col) in zero_as_missing_keys:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace(0, np.nan)

    y = pd.to_numeric(df[target_col], errors="coerce")
    X_df = df.drop(columns=[target_col]).apply(pd.to_numeric, errors="coerce")

    valid_target = ~y.isna()
    X_df = X_df.loc[valid_target].reset_index(drop=True)
    y = y.loc[valid_target].astype(int).reset_index(drop=True)
    _validate_binary_target(y, "diabetes")

    X = X_df.values.astype(float)
    return {
        "name": "Diabetes",
        "X": X,
        "y": y.values,
        "use_nan_aware": True,
        "rows_raw": raw_rows,
        "rows_after_dedup": rows_after_dedup,
        "rows_final": len(X),
        "features": X.shape[1],
        "nan_cells": int(np.isnan(X).sum()),
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

    for col in X_df.columns:
        if not pd.api.types.is_numeric_dtype(X_df[col]):
            X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))

    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    valid = (~X_df.isna().any(axis=1)) & (~y.isna())
    X_df = X_df.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(int).reset_index(drop=True)
    _validate_binary_target(y, "heart_disease")

    X = StandardScaler().fit_transform(X_df.values.astype(float))
    return {
        "name": "Heart Disease",
        "X": X,
        "y": y.values,
        "use_nan_aware": False,
        "categorical_idx": categorical_indices(list(X_df.columns)),
        "rows_raw": raw_rows,
        "rows_after_dedup": rows_after_dedup,
        "rows_final": len(X),
        "features": X.shape[1],
    }


def evaluate_dataset(dataset):
    print(f"\n{'=' * 124}")
    print(f"{dataset['name']} | k={K}")
    print(f"{'=' * 124}")
    if dataset["name"] == "Diabetes":
        print(
            f"Rows raw={dataset['rows_raw']} | dedup={dataset['rows_after_dedup']} | final={dataset['rows_final']} | "
            f"features={dataset['features']} | NaN cells={dataset['nan_cells']} | model=BNB NaN-aware"
        )
    elif dataset["name"] == "Heart Disease":
        print(
            f"Rows raw={dataset['rows_raw']} | dedup={dataset['rows_after_dedup']} | final={dataset['rows_final']} | "
            f"features={dataset['features']} | model=BNB"
        )
    else:
        print(f"Rows raw={dataset['rows_raw']} | final={dataset['rows_final']} | features={dataset['features']} | model=BNB")

    print(f"{'-' * 124}")
    w = 24
    print(f"{'Variant':<20} {'Acc [95% CI]':<{w}} {'Prec [95% CI]':<{w}} {'Rec [95% CI]':<{w}} {'F1 [95% CI]':<{w}} {'AUC [95% CI]':<{w}}")
    print(f"{'-' * 140}")

    results = {}
    for variant_name, method in SPLIT_METHODS.items():
        result = evaluate_variant(
            dataset["X"], dataset["y"], method,
            use_nan_aware=dataset["use_nan_aware"],
            categorical_idx=dataset.get("categorical_idx"),
        )
        results[variant_name] = result
        ci = result["ci"]
        print(
            f"{variant_name:<20} "
            f"{_fmt_metric_ci(result['accuracy'], ci['accuracy']):<{w}} "
            f"{_fmt_metric_ci(result['precision'], ci['precision']):<{w}} "
            f"{_fmt_metric_ci(result['recall'], ci['recall']):<{w}} "
            f"{_fmt_metric_ci(result['f1'], ci['f1']):<{w}} "
            f"{_fmt_metric_ci(result['auc'], ci['auc']):<{w}}"
        )

    print(f"\nConfusion matrices:")
    for variant_name in SPLIT_METHODS.keys():
        cm = results[variant_name]["cm"]
        print(f"  {variant_name:<20} TN={cm['TN']} FP={cm['FP']} FN={cm['FN']} TP={cm['TP']}")

    _print_pairwise_tests_vs_bnb(results, baseline_key="chi2")


def main():
    print("BNB binarization variants across three datasets")
    print(
        "Variants: chi2, mutual_info, gini_gain, information_gain, auc_split, "
        "otsu, median, mean, youden_j, kmeans_midpoint"
    )
    print("Cross-validation folds: k=10")

    datasets = [load_breast_cancer(), load_diabetes(), load_heart_disease()]
    for dataset in datasets:
        evaluate_dataset(dataset)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
