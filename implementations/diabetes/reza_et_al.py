import os
import sys
import copy
import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

_BASE = os.path.dirname(os.path.abspath(__file__))
_ALGO_CANDIDATES = [
    os.path.join(_BASE, "..", "..", "algorithms"),
    os.path.join(_BASE, "..", "algorithms"),
]
for _p in _ALGO_CANDIDATES:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import bnb_nan
from bnb import bootstrap_ci, delong_test, mcnemar_test, paired_bootstrap_diff_ci


# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
_DATASET_CANDIDATES = [
    os.path.join(_BASE, "..", "..", "datasets", "diabetes.csv"),
    os.path.join(_BASE, "..", "datasets", "diabetes.csv"),
]
for _csv_path in _DATASET_CANDIDATES:
    if os.path.isfile(_csv_path):
        break
else:
    _csv_path = _DATASET_CANDIDATES[0]

df = pd.read_csv(_csv_path)

print("=== Raw dataset ===")
print(df.head())
print(df.shape)
print(df.info())

# Normalize possible column-name variants
rename_map = {}
if "Blood Pressure" in df.columns:
    rename_map["Blood Pressure"] = "BloodPressure"
if "Skin Thickness" in df.columns:
    rename_map["Skin Thickness"] = "SkinThickness"
df = df.rename(columns=rename_map)


# ─────────────────────────────────────────────
# 2. CONFIG
# ─────────────────────────────────────────────
RANDOM_STATE = 42
K_VALUES = [5, 10]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

print("\n=== Preprocessing used inside each training fold ===")
print("1. Replace clinical zeros with NaN")
print("2. Mean imputation")
print("3. IQR outlier removal (train fold only)")
print("4. SMOTE (train fold only)")

X_df = df.drop(columns="Outcome").copy()
y = df["Outcome"].values


# ─────────────────────────────────────────────
# 3. PREPROCESSING HELPERS
# ─────────────────────────────────────────────
def replace_zeros_with_nan(df_in, zero_cols):
    df_out = df_in.copy()
    for col in zero_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].replace(0, np.nan)
    return df_out


def fit_iqr_bounds(X_df, columns):
    bounds = {}
    for col in columns:
        q1 = X_df[col].quantile(0.25)
        q3 = X_df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        bounds[col] = (lower, upper)
    return bounds


def apply_iqr_filter(X_df, y_series, bounds):
    mask = pd.Series(True, index=X_df.index)
    for col, (lower, upper) in bounds.items():
        mask &= X_df[col].between(lower, upper)
    return (
        X_df.loc[mask].reset_index(drop=True),
        y_series.loc[mask].reset_index(drop=True),
    )


def fold_preprocess(X_train_df, y_train, X_test_df):
    # 1) zero -> NaN
    X_train_df = replace_zeros_with_nan(X_train_df, ZERO_AS_MISSING)
    X_test_df = replace_zeros_with_nan(X_test_df, ZERO_AS_MISSING)

    # 2) mean imputation fit on train only
    imputer = SimpleImputer(strategy="mean")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train_df),
        columns=X_train_df.columns,
        index=X_train_df.index,
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test_df),
        columns=X_test_df.columns,
        index=X_test_df.index,
    )

    # 3) IQR outlier removal on train only
    y_train_series = pd.Series(y_train, index=X_train_imp.index)
    bounds = fit_iqr_bounds(X_train_imp, list(X_train_imp.columns))
    X_train_filt, y_train_filt = apply_iqr_filter(X_train_imp, y_train_series, bounds)

    # Fallback in case filtering becomes too aggressive
    if len(np.unique(y_train_filt)) < 2 or len(X_train_filt) < 10:
        X_train_filt, y_train_filt = X_train_imp.copy(), y_train_series.copy()

    # 4) SMOTE on train only
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_filt.values, y_train_filt.values)
    # print(f"    After SMOTE: {X_train_bal.shape[0]} samples")

    return X_train_bal, y_train_bal, X_test_imp.values


# ─────────────────────────────────────────────
# 4. BNB HELPERS
# ─────────────────────────────────────────────
def fit_bnb_thresholds(X_train, y_train):
    """
    Best-effort supervised binarization thresholds:
    midpoint of class-wise means per feature.
    """
    cls0 = X_train[y_train == 0]
    cls1 = X_train[y_train == 1]
    thresholds = (cls0.mean(axis=0) + cls1.mean(axis=0)) / 2.0
    return thresholds


def apply_bnb_binarization(X, thresholds):
    return (X >= thresholds).astype(int)


# ─────────────────────────────────────────────
# 5. PAPER KERNELS
# ─────────────────────────────────────────────
def _safe_var(X):
    v = np.var(X, axis=0, ddof=1)
    v[v == 0] = 1.0
    return v


def make_kernel_callable(kind, gamma=0.1, minkowski_p=3, mix_p=0.5):
    def kernel_rbf(X, Y):
        D = cdist(X, Y, metric="sqeuclidean")
        return np.exp(-gamma * D)

    def kernel_cosine_rbf(X, Y):
        D = cdist(X, Y, metric="cosine")
        D = np.nan_to_num(D, nan=0.0, posinf=1.0, neginf=1.0)
        return np.exp(-gamma * D)

    def kernel_minkowski_rbf(X, Y):
        D = cdist(X, Y, metric="minkowski", p=minkowski_p)
        return np.exp(-gamma * D)

    def kernel_cityblock_rbf(X, Y):
        D = cdist(X, Y, metric="cityblock")
        return np.exp(-gamma * D)

    def kernel_mahalanobis_rbf(X, Y):
        cov = np.cov(X, rowvar=False)
        VI = np.linalg.pinv(cov)
        D = cdist(X, Y, metric="mahalanobis", VI=VI)
        return np.exp(-gamma * D)

    def kernel_braycurtis_rbf(X, Y):
        D = cdist(X, Y, metric="braycurtis")
        D = np.nan_to_num(D, nan=0.0, posinf=1.0, neginf=1.0)
        return np.exp(-gamma * D)

    def kernel_canberra_rbf(X, Y):
        D = cdist(X, Y, metric="canberra")
        D = np.nan_to_num(D, nan=0.0, posinf=1.0, neginf=1.0)
        return np.exp(-gamma * D)

    def kernel_chebyshev_rbf(X, Y):
        D = cdist(X, Y, metric="chebyshev")
        return np.exp(-gamma * D)

    def kernel_seuclidean_rbf(X, Y):
        V = _safe_var(X)
        D = cdist(X, Y, metric="seuclidean", V=V)
        return np.exp(-gamma * D)

    def kernel_integrated_1(X, Y):
        return mix_p * kernel_canberra_rbf(X, Y) + (1 - mix_p) * kernel_braycurtis_rbf(X, Y)

    def kernel_integrated_2(X, Y):
        # best-effort stable approximation
        return mix_p * kernel_cosine_rbf(X, Y) + (1 - mix_p) * kernel_chebyshev_rbf(X, Y)

    def kernel_integrated_3(X, Y):
        return mix_p * kernel_minkowski_rbf(X, Y) + (1 - mix_p) * kernel_cosine_rbf(X, Y)

    def kernel_proposed(X, Y):
        return mix_p * kernel_rbf(X, Y) + (1 - mix_p) * kernel_cityblock_rbf(X, Y)

    mapping = {
        "rbf_cosine": kernel_cosine_rbf,
        "rbf_minkowski": kernel_minkowski_rbf,
        "rbf_cityblock": kernel_cityblock_rbf,
        "rbf_mahalanobis": kernel_mahalanobis_rbf,
        "rbf_braycurtis": kernel_braycurtis_rbf,
        "rbf_canberra": kernel_canberra_rbf,
        "rbf_chebyshev": kernel_chebyshev_rbf,
        "rbf_seuclidean": kernel_seuclidean_rbf,
        "integrated_1": kernel_integrated_1,
        "integrated_2": kernel_integrated_2,
        "integrated_3": kernel_integrated_3,
        "proposed_integrated": kernel_proposed,
    }
    return mapping[kind]


# ─────────────────────────────────────────────
# 6. MODELS
# ─────────────────────────────────────────────
def make_models():
    # best-effort defaults because paper does not fully specify these
    C = 1.0
    gamma = 0.1
    degree = 3
    minkowski_p = 3
    mix_p = 0.5

    return {
        "SVM - Linear": SVC(
            kernel="linear",
            C=C,
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "SVM - Polynomial": SVC(
            kernel="poly",
            degree=degree,
            gamma="scale",
            C=C,
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "SVM - RBF": SVC(
            kernel="rbf",
            gamma=gamma,
            C=C,
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "SVM - Sigmoid": SVC(
            kernel="sigmoid",
            gamma=gamma,
            C=C,
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "SVM - RBF Cosine": SVC(
            kernel=make_kernel_callable("rbf_cosine", gamma=gamma, mix_p=mix_p),
            C=C,
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "SVM - RBF Minkowski": SVC(
            kernel=make_kernel_callable(
                "rbf_minkowski",
                gamma=gamma,
                minkowski_p=minkowski_p,
                mix_p=mix_p,
            ),
            C=C,
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "SVM - RBF City Block": SVC(
            kernel=make_kernel_callable("rbf_cityblock", gamma=gamma, mix_p=mix_p),
            C=C,
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "SVM - RBF Mahalanobis": SVC(
            kernel=make_kernel_callable("rbf_mahalanobis", gamma=gamma, mix_p=mix_p),
            C=C,
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "SVM - RBF Bray-Curtis": SVC(
            kernel=make_kernel_callable("rbf_braycurtis", gamma=gamma, mix_p=mix_p),
            C=C,
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "SVM - RBF Canberra": SVC(
            kernel=make_kernel_callable("rbf_canberra", gamma=gamma, mix_p=mix_p),
            C=C,
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "SVM - RBF Chebyshev": SVC(
            kernel=make_kernel_callable("rbf_chebyshev", gamma=gamma, mix_p=mix_p),
            C=C,
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "SVM - RBF standardized ED": SVC(
            kernel=make_kernel_callable("rbf_seuclidean", gamma=gamma, mix_p=mix_p),
            C=C,
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "SVM - Integrated Kernel-1": SVC(
            kernel=make_kernel_callable("integrated_1", gamma=gamma, mix_p=mix_p),
            C=C,
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "SVM - Integrated Kernel-2": SVC(
            kernel=make_kernel_callable("integrated_2", gamma=gamma, mix_p=mix_p),
            C=C,
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "SVM - Integrated Kernel-3": SVC(
            kernel=make_kernel_callable("integrated_3", gamma=gamma, mix_p=mix_p),
            C=C,
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "SVM - Proposed Integrated Kernel": SVC(
            kernel=make_kernel_callable("proposed_integrated", gamma=gamma, mix_p=mix_p),
            C=C,
            probability=True,
            random_state=RANDOM_STATE,
        ),
    }


models = make_models()


# ─────────────────────────────────────────────
# 7. EVALUATION
# ─────────────────────────────────────────────
def evaluate_model(name, model, X_df, y, k_values):
    W = 24

    def _fmt(val, ci_tuple):
        return f"{val:.2f} [{ci_tuple[0]:.2f}, {ci_tuple[1]:.2f}]"

    print(f"\n{'=' * 125}")
    print(f"  {name}")
    print(f"{'=' * 125}")
    print(f"  {'k':<5} {'Accuracy':<{W}} {'Precision':<{W}} {'Recall':<{W}} {'F-Score':<{W}} {'AUC':<{W}}")
    print(f"  {'-' * 122}")

    cm_pred = None
    all_results = {}

    for k in k_values:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)

        y_true_all, y_pred_all, y_prob_all = [], [], []

        for train_idx, test_idx in cv.split(X_df, y):
            X_train_df = X_df.iloc[train_idx].reset_index(drop=True)
            X_test_df = X_df.iloc[test_idx].reset_index(drop=True)
            y_train = y[train_idx]
            y_test = y[test_idx]

            X_train_proc, y_train_proc, X_test_proc = fold_preprocess(X_train_df, y_train, X_test_df)

            model_fold = copy.deepcopy(model)
            model_fold.fit(X_train_proc, y_train_proc)

            y_pred = model_fold.predict(X_test_proc)
            y_prob = model_fold.predict_proba(X_test_proc)[:, 1]

            y_true_all.append(y_test)
            y_pred_all.append(y_pred)
            y_prob_all.append(y_prob)

        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        y_prob_all = np.concatenate(y_prob_all)

        acc = accuracy_score(y_true_all, y_pred_all)
        prec = precision_score(y_true_all, y_pred_all, average="weighted", zero_division=0)
        rec = recall_score(y_true_all, y_pred_all, average="weighted", zero_division=0)
        f1 = f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0)
        auc = roc_auc_score(y_true_all, y_prob_all)

        ci = bootstrap_ci(y_true_all, y_pred_all, y_prob_all)

        print(
            f"  {k:<5} {_fmt(acc, ci['accuracy']):<{W}} {_fmt(prec, ci['precision']):<{W}}"
            f" {_fmt(rec, ci['recall']):<{W}} {_fmt(f1, ci['f1']):<{W}}"
            f" {_fmt(auc, ci['auc']):<{W}}"
        )

        all_results[k] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
            "ci": ci,
            "y_true": y_true_all,
            "y_pred": y_pred_all,
            "y_prob": y_prob_all,
        }

        if k == 10:
            cm_pred = (y_true_all, y_pred_all)

    y_true_cm, y_pred_cm = cm_pred
    tn, fp, fn, tp = confusion_matrix(y_true_cm, y_pred_cm).ravel()
    print("\n  Confusion Matrix (10-fold):")
    print(f"    TN={tn}  FP={fp}")
    print(f"    FN={fn}  TP={tp}")

    return all_results


def evaluate_bnb(X_df, y, k_values):
    W = 24

    def _fmt(val, ci_tuple):
        return f"{val:.2f} [{ci_tuple[0]:.2f}, {ci_tuple[1]:.2f}]"

    print(f"\n{'=' * 125}")
    print("  Bernoulli Naive Bayes")
    print(f"{'=' * 125}")
    print(f"  {'k':<5} {'Accuracy':<{W}} {'Precision':<{W}} {'Recall':<{W}} {'F-Score':<{W}} {'AUC':<{W}}")
    print(f"  {'-' * 122}")

    cm_pred = None
    all_results = {}

    for k in k_values:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)

        y_true_all, y_pred_all, y_prob_all = [], [], []

        for train_idx, test_idx in cv.split(X_df, y):
            X_train_df = X_df.iloc[train_idx].reset_index(drop=True)
            X_test_df = X_df.iloc[test_idx].reset_index(drop=True)
            y_train = y[train_idx]
            y_test = y[test_idx]

            X_train_proc, y_train_proc, X_test_proc = fold_preprocess(X_train_df, y_train, X_test_df)

            thresholds = fit_bnb_thresholds(X_train_proc, y_train_proc)
            X_train_bin = apply_bnb_binarization(X_train_proc, thresholds)
            X_test_bin = apply_bnb_binarization(X_test_proc, thresholds)

            model = BernoulliNB()
            model.fit(X_train_bin, y_train_proc)

            y_pred = model.predict(X_test_bin)
            y_prob = model.predict_proba(X_test_bin)[:, 1]

            y_true_all.append(y_test)
            y_pred_all.append(y_pred)
            y_prob_all.append(y_prob)

        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        y_prob_all = np.concatenate(y_prob_all)

        acc = accuracy_score(y_true_all, y_pred_all)
        prec = precision_score(y_true_all, y_pred_all, average="weighted", zero_division=0)
        rec = recall_score(y_true_all, y_pred_all, average="weighted", zero_division=0)
        f1 = f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0)
        auc = roc_auc_score(y_true_all, y_prob_all)

        ci = bootstrap_ci(y_true_all, y_pred_all, y_prob_all)

        print(
            f"  {k:<5} {_fmt(acc, ci['accuracy']):<{W}} {_fmt(prec, ci['precision']):<{W}}"
            f" {_fmt(rec, ci['recall']):<{W}} {_fmt(f1, ci['f1']):<{W}}"
            f" {_fmt(auc, ci['auc']):<{W}}"
        )

        all_results[k] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
            "ci": ci,
            "y_true": y_true_all,
            "y_pred": y_pred_all,
            "y_prob": y_prob_all,
        }

        if k == 10:
            cm_pred = (y_true_all, y_pred_all)

    y_true_cm, y_pred_cm = cm_pred
    tn, fp, fn, tp = confusion_matrix(y_true_cm, y_pred_cm).ravel()
    print("\n  Confusion Matrix (10-fold):")
    print(f"    TN={tn}  FP={fp}")
    print(f"    FN={fn}  TP={tp}")

    return all_results


# Collect results from all paper models
all_model_results = {}
for name, model in models.items():
    all_model_results[name] = evaluate_model(name, model, X_df, y, K_VALUES)

# Evaluate BNB
bnb_results = evaluate_bnb(X_df, y, K_VALUES)
all_model_results["Bernoulli Naive Bayes"] = bnb_results

# BNB_nan — zeros replaced with NaN, no imputation, no IQR, no SMOTE
X_nan = replace_zeros_with_nan(X_df, ZERO_AS_MISSING).values.astype(float)
bnb_nan_results = bnb_nan.evaluate(X_nan, y, k_values=K_VALUES)
all_model_results["BNB (NaN-aware)"] = bnb_nan_results


# ─────────────────────────────────────────────
# 8. DELONG TEST — pairwise AUC comparison vs BNB (k=10)
# ─────────────────────────────────────────────
print(f"\n{'=' * 90}")
print("  DeLong Test: Pairwise AUC comparison vs Bernoulli Naive Bayes (k=10)")
print(f"{'=' * 90}")
print(f"  {'Model':<35} {'AUC_model':<10} {'AUC_BNB':<10} {'Diff':<10} {'Z':<10} {'p-value':<12} {'Significant?'}")
print(f"  {'-' * 87}")

k_ref = 10
bnb_k10 = bnb_results[k_ref]

compare_models = list(models.keys()) + ['BNB (NaN-aware)']
for name in compare_models:
    other_k10 = all_model_results[name][k_ref]
    result = delong_test(
        bnb_k10["y_true"],
        other_k10["y_prob"],  # Model A
        bnb_k10["y_prob"],    # Model B = BNB
    )
    sig = "Yes" if result["p_value"] < 0.05 else "No"
    print(
        f"  {name:<35} {result['auc_a']:<10.4f} {result['auc_b']:<10.4f} "
        f"{result['diff']:>+10.4f} {result['z_stat']:>10.4f} {result['p_value']:<12.4f} {sig}"
    )

print("\n  Note: p < 0.05 indicates statistically significant difference in AUC.")
print("  'No' means BNB is not significantly different from that model on AUC.")


# ─────────────────────────────────────────────
# 9. McNEMAR TEST — pairwise prediction comparison vs BNB (k=10)
# ─────────────────────────────────────────────
print(f"\n{'=' * 95}")
print("  McNemar Test: Pairwise prediction comparison vs Bernoulli Naive Bayes (k=10)")
print(f"{'=' * 95}")
print(f"  {'Model':<35} {'b (A+B-)':<10} {'c (A-B+)':<10} {'Statistic':<12} {'p-value':<12} {'Method':<8} {'Significant?'}")
print(f"  {'-' * 92}")

for name in compare_models:
    other_k10 = all_model_results[name][k_ref]
    result = mcnemar_test(
        bnb_k10["y_true"],
        other_k10["y_pred"],  # Model A
        bnb_k10["y_pred"],    # Model B = BNB
    )
    sig = "Yes" if result["p_value"] < 0.05 else "No"
    stat_str = f"{result['statistic']:.4f}" if not np.isnan(result["statistic"]) else "—"
    print(
        f"  {name:<35} {result['b']:<10} {result['c']:<10} "
        f"{stat_str:<12} {result['p_value']:<12.4f} {result['method']:<8} {sig}"
    )

print("\n  b = instances where the comparator is correct but BNB is wrong")
print("  c = instances where BNB is correct but the comparator is wrong")
print("  H0: both classifiers make errors on the same instances (b = c)")
print("  p < 0.05 -> significantly different error patterns")


# ─────────────────────────────────────────────
# 10. PAIRED BOOTSTRAP DIFFERENCE CIs — BNB minus comparator (k=10)
# ─────────────────────────────────────────────
print(f"\n{'=' * 112}")
print("  Paired Bootstrap 95% CIs: Bernoulli Naive Bayes minus comparator (k=10)")
print(f"{'=' * 112}")
print(f"  {'Model':<35} {'Acc diff [95% CI]':<24} {'F1 diff [95% CI]':<24} {'AUC diff [95% CI]':<24}")
print(f"  {'-' * 109}")

for name in compare_models:
    other_k10 = all_model_results[name][k_ref]
    diff_result = paired_bootstrap_diff_ci(
        bnb_k10["y_true"],
        bnb_k10["y_pred"],
        bnb_k10["y_prob"],
        other_k10["y_pred"],
        other_k10["y_prob"],
    )

    def _fmt_diff(metric):
        diff = diff_result["observed_diff"][metric]
        lo, hi = diff_result["ci"][metric]
        return f"{diff:+.3f} [{lo:+.3f}, {hi:+.3f}]"

    print(
        f"  {name:<35} "
        f"{_fmt_diff('accuracy'):<24} "
        f"{_fmt_diff('f1'):<24} "
        f"{_fmt_diff('auc'):<24}"
    )

print("\n  Positive differences favor Bernoulli Naive Bayes.")
print("  If a 95% CI includes 0, the paired bootstrap does not show a clear difference on that metric.")


# ─────────────────────────────────────────────
# 11. DIRECT: BNB (NaN-aware) vs BNB (mean imputed) — k=10
# ─────────────────────────────────────────────
bnb_nan_k10 = bnb_nan_results[k_ref]

delong_direct = delong_test(bnb_nan_k10["y_true"], bnb_nan_k10["y_prob"], bnb_k10["y_prob"])
mcnem_direct  = mcnemar_test(bnb_nan_k10["y_true"], bnb_nan_k10["y_pred"], bnb_k10["y_pred"])
boot_direct   = paired_bootstrap_diff_ci(
    bnb_nan_k10["y_true"],
    bnb_nan_k10["y_pred"], bnb_nan_k10["y_prob"],
    bnb_k10["y_pred"],     bnb_k10["y_prob"],
)

print(f"\n{'=' * 80}")
print(f"  Direct comparison: BNB (NaN-aware) vs BNB (mean imputed) — k={k_ref}")
print(f"{'=' * 80}")
print(f"  DeLong:  AUC_nan={delong_direct['auc_a']:.4f}  AUC_imp={delong_direct['auc_b']:.4f}"
      f"  diff={delong_direct['diff']:+.4f}  Z={delong_direct['z_stat']:.4f}"
      f"  p={delong_direct['p_value']:.4f}  sig={'Yes' if delong_direct['p_value'] < 0.05 else 'No'}")
stat_str = f"{mcnem_direct['statistic']:.4f}" if not np.isnan(mcnem_direct["statistic"]) else "—"
print(f"  McNemar: b={mcnem_direct['b']}  c={mcnem_direct['c']}"
      f"  stat={stat_str}  p={mcnem_direct['p_value']:.4f}  method={mcnem_direct['method']}"
      f"  sig={'Yes' if mcnem_direct['p_value'] < 0.05 else 'No'}")

def _fmt_direct(metric):
    diff = boot_direct["observed_diff"][metric]
    lo, hi = boot_direct["ci"][metric]
    return f"{diff:+.3f} [{lo:+.3f}, {hi:+.3f}]"

print(f"  Bootstrap (NaN-aware minus imputed):")
print(f"    Acc  {_fmt_direct('accuracy')}")
print(f"    F1   {_fmt_direct('f1')}")
print(f"    AUC  {_fmt_direct('auc')}")
print(f"  Positive values mean BNB (NaN-aware) outperforms BNB (mean imputed).")


# ─────────────────────────────────────────────
# 12. DELONG TEST — pairwise AUC comparison vs BNB_nan (k=10)
# ─────────────────────────────────────────────
print(f"\n{'=' * 80}")
print("  DeLong Test: Pairwise AUC comparison vs BNB (NaN-aware) (k=10)")
print(f"{'=' * 80}")
print(f"  {'Model':<35} {'AUC_model':<10} {'AUC_BNBnan':<10} {'Diff':<10} {'Z':<10} {'p-value':<12} {'Significant?'}")
print(f"  {'-' * 77}")

compare_models_nan = list(models.keys()) + ['Bernoulli Naive Bayes']
for name in compare_models_nan:
    other_k10 = all_model_results[name][k_ref]
    result = delong_test(
        bnb_nan_k10["y_true"],
        other_k10["y_prob"],
        bnb_nan_k10["y_prob"],
    )
    sig = "Yes" if result["p_value"] < 0.05 else "No"
    print(
        f"  {name:<35} {result['auc_a']:<10.4f} {result['auc_b']:<10.4f} "
        f"{result['diff']:>+10.4f} {result['z_stat']:>10.4f} {result['p_value']:<12.4f} {sig}"
    )

print("\n  Note: p < 0.05 indicates statistically significant difference in AUC.")
print("  'No' means BNB (NaN-aware) is not significantly different from that model.")


# ─────────────────────────────────────────────
# 13. McNEMAR TEST — pairwise error-pattern comparison vs BNB_nan (k=10)
# ─────────────────────────────────────────────
print(f"\n{'=' * 95}")
print("  McNemar Test: Pairwise prediction comparison vs BNB (NaN-aware) (k=10)")
print(f"{'=' * 95}")
print(f"  {'Model':<35} {'b (A+B-)':<10} {'c (A-B+)':<10} {'Statistic':<12} {'p-value':<12} {'Method':<8} {'Significant?'}")
print(f"  {'-' * 92}")

for name in compare_models_nan:
    other_k10 = all_model_results[name][k_ref]
    result = mcnemar_test(
        bnb_nan_k10["y_true"],
        other_k10["y_pred"],
        bnb_nan_k10["y_pred"],
    )
    sig = "Yes" if result["p_value"] < 0.05 else "No"
    stat_str = f"{result['statistic']:.4f}" if not np.isnan(result["statistic"]) else "—"
    print(
        f"  {name:<35} {result['b']:<10} {result['c']:<10} "
        f"{stat_str:<12} {result['p_value']:<12.4f} {result['method']:<8} {sig}"
    )

print("\n  b = instances where the other model is correct but BNB (NaN-aware) is wrong")
print("  c = instances where BNB (NaN-aware) is correct but the other model is wrong")
print("  H0: both classifiers make errors on the same instances (b = c)")
print("  p < 0.05 -> significantly different error patterns")


# ─────────────────────────────────────────────
# 14. PAIRED BOOTSTRAP DIFFERENCE CIs — BNB_nan minus comparator (k=10)
# ─────────────────────────────────────────────
print(f"\n{'=' * 112}")
print("  Paired Bootstrap 95% CIs: BNB (NaN-aware) minus comparator (k=10)")
print(f"{'=' * 112}")
print(f"  {'Model':<35} {'Acc diff [95% CI]':<24} {'F1 diff [95% CI]':<24} {'AUC diff [95% CI]':<24}")
print(f"  {'-' * 109}")

for name in compare_models_nan:
    other_k10 = all_model_results[name][k_ref]
    diff_result = paired_bootstrap_diff_ci(
        bnb_nan_k10["y_true"],
        bnb_nan_k10["y_pred"],
        bnb_nan_k10["y_prob"],
        other_k10["y_pred"],
        other_k10["y_prob"],
    )

    def _fmt_nan_diff(metric):
        diff = diff_result["observed_diff"][metric]
        lo, hi = diff_result["ci"][metric]
        return f"{diff:+.3f} [{lo:+.3f}, {hi:+.3f}]"

    print(
        f"  {name:<35} "
        f"{_fmt_nan_diff('accuracy'):<24} "
        f"{_fmt_nan_diff('f1'):<24} "
        f"{_fmt_nan_diff('auc'):<24}"
    )

print("\n  Positive differences favor BNB (NaN-aware).")
print("  If a 95% CI includes 0, the paired bootstrap does not show a clear difference on that metric.")


# ─────────────────────────────────────────────
# 15. OPTIONAL SUMMARY
# ─────────────────────────────────────────────
rows = []
for name, results in all_model_results.items():
    r = results[10]
    rows.append(
        {
            "Model": name,
            "Accuracy": r["accuracy"],
            "F1": r["f1"],
            "AUC": r["auc"],
        }
    )

summary = pd.DataFrame(rows).sort_values(["AUC", "F1", "Accuracy"], ascending=False)

print(f"\n{'=' * 80}")
print("  Ranking by 10-fold AUC")
print(f"{'=' * 80}")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n=== Done ===")
