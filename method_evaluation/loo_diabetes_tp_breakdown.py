import argparse
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd


SEED = 42
ALPHA = 1.0
ZERO_AS_MISSING_KEYS = {"glucose", "bloodpressure", "skinthickness", "insulin", "bmi"}

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


def load_diabetes_clean():
    path = os.path.join(DATASETS_DIR, "diabetes.csv")
    df = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
    raw_n = len(df)

    target_col = _resolve_column(df.columns, ["Outcome"])
    if target_col is None:
        raise ValueError("diabetes: expected target column 'Outcome'.")

    # Same protocol as the main Pima analysis: physiologically impossible zeros
    # are missing values, handled by the NaN-aware model (records are kept).
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if col != target_col and _canon(col) in ZERO_AS_MISSING_KEYS:
            df[col] = df[col].replace(0, np.nan)

    cleaned = df.dropna(subset=[target_col]).reset_index(drop=True)
    y = cleaned[target_col].astype(int).to_numpy()
    X_df = cleaned.drop(columns=[target_col]).copy()
    _validate_binary_target(y, "diabetes")

    return cleaned, X_df, y, raw_n


def chi2_threshold(x_col, y):
    """Chi2 threshold on the observed values, with Yates' correction as in the models."""
    x_col = np.asarray(x_col, dtype=float)
    y = np.asarray(y, dtype=int)
    observed = ~np.isnan(x_col)
    x_col, y = x_col[observed], y[observed]
    if len(x_col) == 0:
        return 0.0

    uniq, inv = np.unique(x_col, return_inverse=True)
    if len(uniq) == 1:
        return float(uniq[0])
    if len(np.unique(y)) < 2:
        return float(uniq[0])

    pos = (y == 1).astype(np.int64)
    neg = 1 - pos

    pos_per = np.bincount(inv, weights=pos, minlength=len(uniq)).astype(float)
    neg_per = np.bincount(inv, weights=neg, minlength=len(uniq)).astype(float)

    low_pos = np.cumsum(pos_per)
    low_neg = np.cumsum(neg_per)
    total_pos = low_pos[-1]
    total_neg = low_neg[-1]
    high_pos = total_pos - low_pos
    high_neg = total_neg - low_neg

    a = low_neg
    b = low_pos
    c = high_neg
    d = high_pos

    n = a + b + c + d
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    chi2 = np.full_like(denom, -np.inf, dtype=float)
    valid = denom > 0
    diff = np.maximum(np.abs(a * d - b * c) - n / 2.0, 0.0)
    chi2[valid] = n[valid] * diff[valid] ** 2 / denom[valid]

    if not np.any(np.isfinite(chi2)):
        return float(uniq[0])
    return float(uniq[int(np.nanargmax(chi2))])


def binarize(X, thresholds):
    X = np.asarray(X, dtype=float)
    return np.where(np.isnan(X), np.nan, (X > thresholds).astype(float))


def fit_bernoulli_nb_params(X_train_bin, y_train, alpha=ALPHA):
    """NaN-aware estimates: each feature uses only the records where it is observed."""
    X_train_bin = np.asarray(X_train_bin, dtype=float)
    y_train = np.asarray(y_train, dtype=int)

    params = {}
    n_train = len(y_train)
    for c in [0, 1]:
        mask = y_train == c
        Xc = X_train_bin[mask]
        n_c = int(mask.sum())
        n_valid = (~np.isnan(Xc)).sum(axis=0).astype(int)
        count1 = np.nansum(Xc, axis=0).astype(int)
        count0 = n_valid - count1

        p1 = (count1 + alpha) / (n_valid + 2.0 * alpha)
        p0 = (count0 + alpha) / (n_valid + 2.0 * alpha)

        params[c] = {
            "n_class": n_c,
            "n_valid": n_valid,
            "prior": n_c / n_train,
            "count1": count1,
            "count0": count0,
            "p1": p1,
            "p0": p0,
        }
    return params


def predict_from_params(x_bin, params):
    x_bin = np.asarray(x_bin, dtype=float)
    observed = ~np.isnan(x_bin)
    log_joint = {}
    for c in [0, 1]:
        prior = params[c]["prior"]
        if prior <= 0:
            log_joint[c] = float("-inf")
            continue

        p1 = np.clip(params[c]["p1"], 1e-15, 1 - 1e-15)
        p0 = np.clip(params[c]["p0"], 1e-15, 1 - 1e-15)
        # Missing features are skipped (likelihood ratio of 1).
        ll_terms = np.where(x_bin == 1, np.log(p1), np.log(p0))
        log_joint[c] = float(np.log(prior) + ll_terms[observed].sum())

    mx = max(log_joint[0], log_joint[1])
    z = np.exp(log_joint[0] - mx) + np.exp(log_joint[1] - mx)
    prob0 = float(np.exp(log_joint[0] - mx) / z)
    prob1 = float(np.exp(log_joint[1] - mx) / z)
    pred = int(prob1 >= prob0)
    return pred, prob0, prob1, log_joint


def loo_predictions(X, y, alpha=ALPHA):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    n, p = X.shape

    out = []
    all_idx = np.arange(n)
    for i in range(n):
        train_idx = all_idx[all_idx != i]
        X_train = X[train_idx]
        y_train = y[train_idx]
        x_test = X[i]

        thresholds = np.array([chi2_threshold(X_train[:, j], y_train) for j in range(p)], dtype=float)
        X_train_bin = binarize(X_train, thresholds)
        x_test_bin = binarize(x_test, thresholds)

        params = fit_bernoulli_nb_params(X_train_bin, y_train, alpha=alpha)
        pred, prob0, prob1, log_joint = predict_from_params(x_test_bin, params)

        out.append(
            {
                "sample_idx": i,
                "y_true": int(y[i]),
                "y_pred": int(pred),
                "prob0": prob0,
                "prob1": prob1,
                "log_joint0": log_joint[0],
                "log_joint1": log_joint[1],
                "thresholds": thresholds,
                "x_test_bin": x_test_bin,
                "params": params,
            }
        )
    return out


def _fmt(v):
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6f}"
    return str(v)


def _default_excel_path():
    out_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "loo_diabetes_tp_breakdown.xlsx")


def _row_has_missing_like_values(row, feature_names):
    for feat in feature_names:
        val = row[feat]
        if pd.isna(val):
            return True
        if _canon(feat) in ZERO_AS_MISSING_KEYS and float(val) == 0.0:
            return True
    return False


def build_feature_tables(feature_names, x_raw, x_bin, thresholds, params):
    summary_rows = []
    like_rows = []
    prior_diabetes = float(params[1]["prior"])
    prior_non_diabetes = float(params[0]["prior"])

    for j, feat in enumerate(feature_names):
        v = int(x_bin[j])

        y1_above = int(params[1]["count1"][j])  # diabetes, x > t
        y1_under = int(params[1]["count0"][j])  # diabetes, x <= t
        y0_above = int(params[0]["count1"][j])  # non-diabetes, x > t
        y0_under = int(params[0]["count0"][j])  # non-diabetes, x <= t
        # Class totals over the records where this feature is observed.
        c0_n = int(params[0]["n_valid"][j])
        c1_n = int(params[1]["n_valid"][j])

        like_d_under = float(params[1]["p0"][j])
        like_d_above = float(params[1]["p1"][j])
        like_nd_under = float(params[0]["p0"][j])
        like_nd_above = float(params[0]["p1"][j])

        like_d_selected = like_d_above if v == 1 else like_d_under
        like_nd_selected = like_nd_above if v == 1 else like_nd_under

        summary_rows.append(
            {
                "Features": feat,
                "Threshold": float(thresholds[j]),
                "Diabetes patients Under threshold": y1_under,
                "Diabetes patients Above Threshold": y1_above,
                "Diabetes patients Total": c1_n,
                "Non Diabetes patients Under threshold": y0_under,
                "Non Diabetes patients Above Threshold": y0_above,
                "Non Diabetes patients Total": c0_n,
                "Prior probability for diabetes": prior_diabetes,
                "Prior probability for non diabetes": prior_non_diabetes,
                "Sample": float(x_raw[j]),
                "Binarized sample": v,
                "Likelihood Diabetes Under threshold": like_d_under,
                "Likelihood Diabetes Above Threshold": like_d_above,
                "Likelihood Non Diabetes Under threshold": like_nd_under,
                "Likelihood Non Diabetes Above Threshold": like_nd_above,
                "Selected Likelihood Diabetes": like_d_selected,
                "Selected Likelihood Non Diabetes": like_nd_selected,
                "Posterior Diabetes (likelihood*prior)": prior_diabetes * like_d_selected,
                "Posterior Non Diabetes (likelihood*prior)": prior_non_diabetes * like_nd_selected,
                "Final probability Diabetes": np.nan,
                "Final probability Non Diabetes": np.nan,
            }
        )

        c0_count_v = int(y0_above if v == 1 else y0_under)
        c1_count_v = int(y1_above if v == 1 else y1_under)
        p0 = float(like_nd_selected)
        p1 = float(like_d_selected)
        like_rows.append(
            {
                "Feature": feat,
                "Sample Bin (x>t)": v,
                "Count in y=0 for sample bin": c0_count_v,
                "Count in y=1 for sample bin": c1_count_v,
                "P(x_j=v|y=0) smoothed": p0,
                "P(x_j=v|y=1) smoothed": p1,
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(like_rows)


def _write_excel_file(path, sample_row, summary_rows, like_rows, meta_rows):
    with pd.ExcelWriter(path) as writer:
        sample_row.to_excel(writer, sheet_name="Sample", index=False)
        summary_rows.to_excel(writer, sheet_name="Count_Table", index=False)
        like_rows.to_excel(writer, sheet_name="Likelihood_Terms", index=False)
        pd.DataFrame(meta_rows).to_excel(writer, sheet_name="Summary", index=False)


def save_report_to_excel(excel_path, sample_row, summary_rows, like_rows, meta_rows):
    out_dir = os.path.dirname(os.path.abspath(excel_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        _write_excel_file(excel_path, sample_row, summary_rows, like_rows, meta_rows)
        return excel_path
    except PermissionError:
        base, ext = os.path.splitext(excel_path)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = f"{base}_{ts}{ext or '.xlsx'}"
        _write_excel_file(fallback, sample_row, summary_rows, like_rows, meta_rows)
        return fallback


def print_tp_report(cleaned_df, X_df, y, loo_out, seed=SEED, excel_path=None, sample_idx=None):
    rng = np.random.default_rng(seed)
    feature_names = list(X_df.columns)
    tps_all = [r for r in loo_out if r["y_true"] == 1 and r["y_pred"] == 1]
    tps = []
    for r in tps_all:
        i = r["sample_idx"]
        row = cleaned_df.iloc[i]
        if not _row_has_missing_like_values(row, feature_names):
            tps.append(r)

    if not tps:
        raise RuntimeError("No true positives found under leave-one-out on the cleaned diabetes dataset.")

    if sample_idx is None:
        picked = tps[int(rng.integers(0, len(tps)))]
    else:
        picked = loo_out[sample_idx]
    i = picked["sample_idx"]
    x_raw = X_df.iloc[i].to_numpy(dtype=float)
    x_bin = picked["x_test_bin"]
    thresholds = picked["thresholds"]
    params = picked["params"]

    summary_rows, like_rows = build_feature_tables(feature_names, x_raw, x_bin, thresholds, params)

    prior0 = float(params[0]["prior"])
    prior1 = float(params[1]["prior"])
    log_joint0 = float(picked["log_joint0"])
    log_joint1 = float(picked["log_joint1"])
    score0 = float(np.exp(log_joint0))
    score1 = float(np.exp(log_joint1))
    total_mult_0 = float(prior0 * np.prod(summary_rows["Selected Likelihood Non Diabetes"].to_numpy(dtype=float)))
    total_mult_1 = float(prior1 * np.prod(summary_rows["Selected Likelihood Diabetes"].to_numpy(dtype=float)))

    print("=" * 130)
    print("Diabetes | Leave-One-Out Bernoulli NB | Random True Positive Detailed Breakdown")
    print("=" * 130)
    print(f"Random seed for TP pick: {seed}")
    print(f"Total cleaned rows used in LOO: {len(y)}")
    print(f"Total TP candidates: {len(tps)}")
    print("-" * 130)
    print(f"Selected sample (cleaned index): {i}")
    print(f"Ground truth y: {picked['y_true']} | Predicted y: {picked['y_pred']}")
    print(f"P(y=0|x): {picked['prob0']:.6f}")
    print(f"P(y=1|x): {picked['prob1']:.6f}")
    print("-" * 130)
    print("Selected sample raw values:")
    sample_row = cleaned_df.iloc[[i]][X_df.columns.tolist() + [_resolve_column(cleaned_df.columns, ['Outcome'])]]
    print(sample_row.to_string(index=False))
    print("-" * 130)
    print("Class priors and joint terms:")
    print(f"Prior P(y=0): {prior0:.6f}")
    print(f"Prior P(y=1): {prior1:.6f}")
    print(f"Unnormalized score y=0 (prior x likelihood product): {score0:.12e}")
    print(f"Unnormalized score y=1 (prior x likelihood product): {score1:.12e}")
    print(f"Total multiplication y=0 = prior * product(selected likelihoods): {total_mult_0:.12e}")
    print(f"Total multiplication y=1 = prior * product(selected likelihoods): {total_mult_1:.12e}")
    print(
        "Posterior check: P(y=1|x) = score1 / (score0 + score1) "
        f"= {score1:.12e} / {(score0 + score1):.12e} = {picked['prob1']:.6f}"
    )
    print("-" * 130)
    summary_export = summary_rows.copy()
    final_row = {col: np.nan for col in summary_export.columns}
    final_row["Features"] = "FINAL"
    final_row["Posterior Diabetes (likelihood*prior)"] = total_mult_1
    final_row["Posterior Non Diabetes (likelihood*prior)"] = total_mult_0
    final_row["Final probability Diabetes"] = picked["prob1"]
    final_row["Final probability Non Diabetes"] = picked["prob0"]
    summary_export = pd.concat([summary_export, pd.DataFrame([final_row])], ignore_index=True)

    print("Per-feature count table (matching your spreadsheet layout):")
    print(summary_export.to_string(index=False, formatters={col: _fmt for col in summary_export.columns}))
    print("-" * 130)
    print("Likelihood terms used for this sample:")
    print(like_rows.to_string(index=False, formatters={col: _fmt for col in like_rows.columns}))
    print("=" * 130)

    if excel_path:
        meta_rows = [
            {"Metric": "Random seed for TP pick", "Value": seed},
            {"Metric": "Selected sample index", "Value": i},
            {"Metric": "Ground truth y", "Value": picked["y_true"]},
            {"Metric": "Predicted y", "Value": picked["y_pred"]},
            {"Metric": "P(y=0|x)", "Value": picked["prob0"]},
            {"Metric": "P(y=1|x)", "Value": picked["prob1"]},
            {"Metric": "Prior P(y=0)", "Value": prior0},
            {"Metric": "Prior P(y=1)", "Value": prior1},
            {"Metric": "Unnormalized score y=0", "Value": score0},
            {"Metric": "Unnormalized score y=1", "Value": score1},
            {"Metric": "Total multiplication y=0 (prior*product likelihoods)", "Value": total_mult_0},
            {"Metric": "Total multiplication y=1 (prior*product likelihoods)", "Value": total_mult_1},
            {"Metric": "Final P(y=0|x)", "Value": picked["prob0"]},
            {"Metric": "Final P(y=1|x)", "Value": picked["prob1"]},
            {"Metric": "Total TP candidates (all)", "Value": len(tps_all)},
            {"Metric": "Total TP candidates (non-missing sample)", "Value": len(tps)},
            {"Metric": "Total cleaned rows used in LOO", "Value": len(y)},
        ]
        saved_path = save_report_to_excel(excel_path, sample_row, summary_export, like_rows, meta_rows)
        print(f"Saved Excel report: {saved_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Leave-one-out classification on Diabetes using chi2-thresholded Bernoulli NB, "
            "then print one random true-positive sample with full probability breakdown."
        )
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for TP sample selection (default: 42).")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Laplace smoothing alpha (default: 1.0).")
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=None,
        help="Report this record instead of a random true positive (index 43 is the patient in the paper).",
    )
    parser.add_argument(
        "--excel_path",
        type=str,
        default=_default_excel_path(),
        help="Output .xlsx path for the report (default: method_evaluation/outputs/loo_diabetes_tp_breakdown.xlsx).",
    )
    args = parser.parse_args()

    if args.alpha <= 0:
        raise ValueError("--alpha must be > 0.")

    cleaned_df, X_df, y, raw_n = load_diabetes_clean()
    print(f"Rows raw: {raw_n} | Rows after cleaning: {len(y)}")
    print(f"Class counts after cleaning -> y=0: {(y == 0).sum()} | y=1: {(y == 1).sum()}")
    print()

    loo_out = loo_predictions(X_df.to_numpy(dtype=float), y, alpha=args.alpha)
    print_tp_report(
        cleaned_df, X_df, y, loo_out, seed=args.seed, excel_path=args.excel_path, sample_idx=args.sample_idx
    )
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
