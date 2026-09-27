import argparse
import os
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


SEED = 42
BOOTSTRAP_N = 1000
BOOTSTRAP_FRACTION = 0.8

HEART_CATEGORICAL_FEATURES = [
    ("ExerciseAngina", "Exercise Angina"),
    ("ChestPainType", "Chest Pain Type"),
    ("RestingECG", "Resting ECG"),
    ("Sex", "Sex"),
    ("ST_Slope", "ST Slope"),
]

HEART_CATEGORY_NAME_MAP = {
    "ExerciseAngina": {"N": "No", "Y": "Yes"},
    "Sex": {"F": "Female", "M": "Male"},
}


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")


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


def _mean_threshold(x):
    x = np.asarray(x, dtype=float)

    if len(x) == 0:
        return float("nan")
    return float(np.mean(x))


def _conditional_probs(x, y, threshold):
    low = x < threshold
    high = x >= threshold

    # If strict split creates an empty side (common for binary/categorical
    # features with threshold at the minimum), fall back to <= / >.
    if (not np.any(low)) or (not np.any(high)):
        low = x <= threshold
        high = x > threshold

    p_low = float(np.mean(y[low])) if np.any(low) else float("nan")
    p_high = float(np.mean(y[high])) if np.any(high) else float("nan")
    delta_p = float(p_high - p_low) if np.isfinite(p_low) and np.isfinite(p_high) else float("nan")
    return p_low, p_high, delta_p


def _percentile_ci(values, ci=95):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    lo = (100 - ci) / 2.0
    hi = 100 - lo
    return (float(np.percentile(arr, lo)), float(np.percentile(arr, hi)))


def _bootstrap_feature_cis(x, y, n_boot=BOOTSTRAP_N, seed=SEED, boot_frac=BOOTSTRAP_FRACTION):
    rng = np.random.default_rng(seed)
    n = len(x)
    if n == 0:
        nan_ci = (float("nan"), float("nan"))
        return nan_ci, nan_ci, nan_ci, nan_ci
    if not (0 < boot_frac <= 1):
        raise ValueError(f"boot_frac must be in (0, 1], got {boot_frac}")

    thresholds = []
    p_lows = []
    p_highs = []
    deltas = []

    subset_n = max(2, int(np.floor(boot_frac * n)))
    subset_n = min(subset_n, n)

    for _ in range(n_boot):
        if subset_n < n:
            subset_idx = rng.choice(n, size=subset_n, replace=False)
        else:
            subset_idx = np.arange(n)

        # Bootstrap from the selected subset (sample with replacement).
        boot_local_idx = rng.integers(0, len(subset_idx), size=len(subset_idx))
        idx = subset_idx[boot_local_idx]

        xb = x[idx]
        yb = y[idx]
        t = _mean_threshold(xb)
        p_low, p_high, delta = _conditional_probs(xb, yb, t)
        thresholds.append(t)
        p_lows.append(p_low)
        p_highs.append(p_high)
        deltas.append(delta)

    return (
        _percentile_ci(thresholds, ci=95),
        _percentile_ci(p_lows, ci=95),
        _percentile_ci(p_highs, ci=95),
        _percentile_ci(deltas, ci=95),
    )


def _category_probability(x_codes, y, category_code):
    mask = x_codes == category_code
    return float(np.mean(y[mask])) if np.any(mask) else float("nan")


def _bootstrap_category_prob_ci(x_codes, y, category_code, n_boot=BOOTSTRAP_N, seed=SEED, boot_frac=BOOTSTRAP_FRACTION):
    rng = np.random.default_rng(seed)
    n = len(x_codes)
    if n == 0:
        return (float("nan"), float("nan"))
    if not (0 < boot_frac <= 1):
        raise ValueError(f"boot_frac must be in (0, 1], got {boot_frac}")

    probs = []
    subset_n = max(2, int(np.floor(boot_frac * n)))
    subset_n = min(subset_n, n)

    for _ in range(n_boot):
        if subset_n < n:
            subset_idx = rng.choice(n, size=subset_n, replace=False)
        else:
            subset_idx = np.arange(n)

        boot_local_idx = rng.integers(0, len(subset_idx), size=len(subset_idx))
        idx = subset_idx[boot_local_idx]
        probs.append(_category_probability(x_codes[idx], y[idx], category_code))

    return _percentile_ci(probs, ci=95)


def _fmt(v):
    return f"{v:.3f}" if np.isfinite(v) else "nan"


def _fmt_with_ci(v, ci_tuple):
    return f"{_fmt(v)} [{_fmt(ci_tuple[0])}, {_fmt(ci_tuple[1])}]"


def _feature_summary_table(X_df, y, n_boot=BOOTSTRAP_N, seed=SEED, boot_frac=BOOTSTRAP_FRACTION):
    rows = []
    for i, feature in enumerate(X_df.columns):
        x = pd.to_numeric(X_df[feature], errors="coerce").to_numpy(dtype=float)
        valid = ~np.isnan(x)
        xv = x[valid]
        yv = y[valid]

        if len(xv) == 0:
            rows.append(
                {
                    "Feature": feature,
                    "Minimum": np.nan,
                    "Maximum": np.nan,
                    "Threshold [95% CI]": "nan [nan, nan]",
                    "P(y=1 | x<thr) [95% CI]": "nan [nan, nan]",
                    "P(y=1 | x>=thr) [95% CI]": "nan [nan, nan]",
                    "Delta P [95% CI]": "nan [nan, nan]",
                }
            )
            continue

        threshold = _mean_threshold(xv)
        p_low, p_high, delta_p = _conditional_probs(xv, yv, threshold)
        thr_ci, p_low_ci, p_high_ci, delta_ci = _bootstrap_feature_cis(
            xv, yv, n_boot=n_boot, seed=seed + i, boot_frac=boot_frac
        )

        rows.append(
            {
                "Feature": feature,
                "Minimum": float(np.min(xv)),
                "Maximum": float(np.max(xv)),
                "Threshold [95% CI]": _fmt_with_ci(threshold, thr_ci),
                "P(y=1 | x<thr) [95% CI]": _fmt_with_ci(p_low, p_low_ci),
                "P(y=1 | x>=thr) [95% CI]": _fmt_with_ci(p_high, p_high_ci),
                "Delta P [95% CI]": _fmt_with_ci(delta_p, delta_ci),
            }
        )

    table = pd.DataFrame(rows)
    table["Minimum"] = table["Minimum"].map(_fmt)
    table["Maximum"] = table["Maximum"].map(_fmt)
    return table


def _clean_heart_disease_df():
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

    return df, target_col


def _heart_categorical_summary_table(n_boot=BOOTSTRAP_N, seed=SEED, boot_frac=BOOTSTRAP_FRACTION):
    df, target_col = _clean_heart_disease_df()

    y = pd.to_numeric(df[target_col], errors="coerce")
    valid_target = ~y.isna()
    df = df.loc[valid_target].reset_index(drop=True)
    y = y.loc[valid_target].astype(int).reset_index(drop=True).to_numpy()
    _validate_binary_target(y, "heart_disease")

    rows = []
    for feature_idx, (feature_col, feature_name) in enumerate(HEART_CATEGORICAL_FEATURES):
        if feature_col not in df.columns:
            continue

        x_raw = df[feature_col].astype(str).to_numpy()
        encoder = LabelEncoder()
        x_codes = encoder.fit_transform(x_raw)
        code_to_raw = {int(i): cls for i, cls in enumerate(encoder.classes_)}

        for code in sorted(np.unique(x_codes)):
            p = _category_probability(x_codes, y, int(code))
            p_ci = _bootstrap_category_prob_ci(
                x_codes,
                y,
                int(code),
                n_boot=n_boot,
                seed=seed + 100 * feature_idx + int(code),
                boot_frac=boot_frac,
            )
            raw_category = code_to_raw[int(code)]
            pretty_category = HEART_CATEGORY_NAME_MAP.get(feature_col, {}).get(raw_category, raw_category)
            rows.append(
                {
                    "Feature": feature_name,
                    "Category Code": int(code),
                    "Category Name": pretty_category,
                    "Probability": float(p),
                    "Probability [95% CI]": _fmt_with_ci(p, p_ci),
                }
            )

    table = pd.DataFrame(rows)
    if len(table) == 0:
        return table

    feature_order = [display_name for _, display_name in HEART_CATEGORICAL_FEATURES]
    table["Feature"] = pd.Categorical(table["Feature"], categories=feature_order, ordered=True)
    table = table.sort_values(["Feature", "Probability"], ascending=[True, False], kind="mergesort").reset_index(drop=True)
    table["Feature"] = table["Feature"].astype(str)
    table["Probability"] = table["Probability"].map(_fmt)
    return table


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
        for id_col in ["id", "ID", "Sample code number"]:
            if id_col in X_df.columns:
                X_df = X_df.drop(columns=[id_col])
    else:
        raise ValueError("breast_cancer: expected 'diagnosis' or 'Class' target column.")

    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    valid_target = ~y.isna()
    X_df = X_df.loc[valid_target].reset_index(drop=True)
    y = y.loc[valid_target].astype(int).reset_index(drop=True).to_numpy()
    _validate_binary_target(y, "breast_cancer")
    return "Breast Cancer", X_df, y


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
    X_df = X_df.loc[valid_target].reset_index(drop=True)
    y = y.loc[valid_target].astype(int).reset_index(drop=True).to_numpy()
    _validate_binary_target(y, "diabetes")
    return "Diabetes", X_df, y


def load_heart_disease():
    df, target_col = _clean_heart_disease_df()

    y = pd.to_numeric(df[target_col], errors="coerce")
    X_df = df.drop(columns=[target_col]).copy()
    for col in X_df.columns:
        if not pd.api.types.is_numeric_dtype(X_df[col]):
            X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))

    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    valid_target = ~y.isna()
    X_df = X_df.loc[valid_target].reset_index(drop=True)
    y = y.loc[valid_target].astype(int).reset_index(drop=True).to_numpy()
    _validate_binary_target(y, "heart_disease")
    return "Heart Disease", X_df, y


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Print per-feature summary tables using mean thresholds on the whole dataset, "
            "with bootstrap 95% CIs for threshold, conditional probabilities, and delta P."
        )
    )
    parser.add_argument("--n_boot", type=int, default=BOOTSTRAP_N, help="Bootstrap iterations (default: 1000)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42)")
    parser.add_argument(
        "--boot_frac",
        type=float,
        default=BOOTSTRAP_FRACTION,
        help="Fraction of rows used per bootstrap iteration before resampling (default: 0.8)",
    )
    args = parser.parse_args()
    if not (0 < args.boot_frac <= 1):
        raise ValueError(f"--boot_frac must be in (0, 1], got {args.boot_frac}")

    loaders = [load_breast_cancer, load_diabetes, load_heart_disease]
    for loader in loaders:
        dataset_name, X_df, y = loader()
        print(f"\n{'=' * 140}")
        print(
            f"{dataset_name} | Mean threshold on whole dataset | "
            f"Bootstrap n={args.n_boot}, fraction={args.boot_frac:.2f}"
        )
        print(f"{'=' * 140}")
        table = _feature_summary_table(
            X_df,
            y,
            n_boot=args.n_boot,
            seed=args.seed,
            boot_frac=args.boot_frac,
        )
        print(table.to_string(index=False))

    cat_table = _heart_categorical_summary_table(n_boot=args.n_boot, seed=args.seed, boot_frac=args.boot_frac)
    print(f"\n{'=' * 140}")
    print(
        "Heart Disease | Categorical feature probabilities | "
        f"Bootstrap n={args.n_boot}, fraction={args.boot_frac:.2f}"
    )
    print(f"{'=' * 140}")
    print(cat_table.to_string(index=False))

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
