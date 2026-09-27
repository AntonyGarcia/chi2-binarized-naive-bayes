import os
import sys
import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

_BASE = os.path.dirname(os.path.abspath(__file__))
_ALGO_CANDIDATES = [
    os.path.join(_BASE, "..", "..", "algorithms"),
    os.path.join(_BASE, "..", "algorithms"),
]
for _p in _ALGO_CANDIDATES:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import bnb
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

# ─────────────────────────────────────────────
# 2. PREPROCESSING  (paper-aligned)
#    • Zero values in Glucose and BMI treated as invalid/missing
#      and replaced by feature mean
#    • Keep only the 4 features used in the paper:
#         Pregnancies, Glucose, BMI, Age
#    • Min-max normalization to [0, 1]
# ─────────────────────────────────────────────
FEATURES = ['Pregnancies', 'Glucose', 'BMI', 'Age']
TARGET = 'Outcome'

df_clean = df.copy()

# Mean imputation for invalid zeros in paper-used columns
for col in ['Glucose', 'BMI']:
    df_clean.loc[df_clean[col] == 0, col] = np.nan
    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

# Keep only paper-selected features
X = df_clean[FEATURES].copy().values.astype(float)
y = df_clean[TARGET].values

# Min-max normalization
X_min = X.min(axis=0)
X_max = X.max(axis=0)
den = (X_max - X_min)
den[den == 0] = 1.0
X = (X - X_min) / den

print("\n=== After paper-aligned preprocessing ===")
print(f"Selected features : {FEATURES}")
print(f"Instances retained: {len(df_clean)}")
print(pd.Series(y).value_counts().rename({0: 'Negative', 1: 'Positive'}))

# ─────────────────────────────────────────────
# 3. MODELS  (Pranto et al., 2020)
#    Keep BNB for reference only.
# ─────────────────────────────────────────────
models = {
    "Gaussian Naive Bayes": GaussianNB(),

    "Decision Tree": DecisionTreeClassifier(
        criterion='entropy',
        max_depth=2,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=800,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=16,
        metric='euclidean'
    ),
}

K_VALUES = [5, 10, 15, 20]

# ─────────────────────────────────────────────
# 4. EVALUATION
# ─────────────────────────────────────────────
def evaluate_model(name, model, X, y, k_values):
    """
    Evaluate model and return results dict including predictions for paired comparison.
    """
    W = 24  # column width per metric

    def _fmt(val, ci_tuple):
        return f"{val:.2f} [{ci_tuple[0]:.2f}, {ci_tuple[1]:.2f}]"

    print(f"\n{'='*125}")
    print(f"  {name}")
    print(f"{'='*125}")
    print(f"  {'k':<5} {'Accuracy':<{W}} {'Precision':<{W}} {'Recall':<{W}} {'F-Score':<{W}} {'AUC':<{W}}")
    print(f"  {'-'*122}")

    cm_pred = None
    all_results = {}

    for k in k_values:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        y_true_all, y_pred_all, y_prob_all = [], [], []
        for train_idx, test_idx in cv.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            y_true_all.append(y[test_idx])
            y_pred_all.append(model.predict(X[test_idx]))
            y_prob_all.append(model.predict_proba(X[test_idx])[:, 1])

        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        y_prob_all = np.concatenate(y_prob_all)

        acc = accuracy_score(y_true_all, y_pred_all)
        prec = precision_score(y_true_all, y_pred_all, average='weighted')
        rec = recall_score(y_true_all, y_pred_all, average='weighted')
        f1 = f1_score(y_true_all, y_pred_all, average='weighted')
        auc = roc_auc_score(y_true_all, y_prob_all)

        ci = bootstrap_ci(y_true_all, y_pred_all, y_prob_all)

        print(
            f"  {k:<5} {_fmt(acc, ci['accuracy']):<{W}} {_fmt(prec, ci['precision']):<{W}}"
            f" {_fmt(rec, ci['recall']):<{W}} {_fmt(f1, ci['f1']):<{W}}"
            f" {_fmt(auc, ci['auc']):<{W}}"
        )

        all_results[k] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'auc': auc,
            'ci': ci,
            'y_true': y_true_all,
            'y_pred': y_pred_all,
            'y_prob': y_prob_all,
        }

        if k == 10:
            cm_pred = (y_true_all, y_pred_all)

    y_true_cm, y_pred_cm = cm_pred
    tn, fp, fn, tp = confusion_matrix(y_true_cm, y_pred_cm).ravel()
    print(f"\n  Confusion Matrix (10-fold):")
    print(f"    TN={tn}  FP={fp}")
    print(f"    FN={fn}  TP={tp}")

    return all_results

# Collect results from all models for paired comparison
all_model_results = {}
for name, model in models.items():
    all_model_results[name] = evaluate_model(name, model, X, y, K_VALUES)

# BNB - binarization is handled inside bnb.py
bnb_results = bnb.evaluate(X, y, k_values=K_VALUES)
all_model_results['Bernoulli Naive Bayes'] = bnb_results

# BNB_nan — keep zeros as NaN (no imputation), min-max normalize observed values
df_nan = df.copy()
for col in ['Glucose', 'BMI']:
    df_nan.loc[df_nan[col] == 0, col] = np.nan
X_nan = df_nan[FEATURES].copy().values.astype(float)
# Min-max normalize using only observed (non-NaN) values
X_nan_min = np.nanmin(X_nan, axis=0)
X_nan_max = np.nanmax(X_nan, axis=0)
den_nan = (X_nan_max - X_nan_min)
den_nan[den_nan == 0] = 1.0
X_nan = (X_nan - X_nan_min) / den_nan
y_nan = df_nan[TARGET].values

bnb_nan_results = bnb_nan.evaluate(X_nan, y_nan, k_values=K_VALUES)
all_model_results['BNB (NaN-aware)'] = bnb_nan_results

# ─────────────────────────────────────────────
# 5. DELONG TEST — pairwise AUC comparison vs BNB (k=10)
# ─────────────────────────────────────────────
print(f"\n{'='*80}")
print("  DeLong Test: Pairwise AUC comparison vs Bernoulli Naive Bayes (k=10)")
print(f"{'='*80}")
print(f"  {'Model':<25} {'AUC_model':<10} {'AUC_BNB':<10} {'Diff':<10} {'Z':<10} {'p-value':<12} {'Significant?'}")
print(f"  {'-'*77}")

k_ref = 10
bnb_k10 = bnb_results[k_ref]
compare_models = list(models.keys()) + ['BNB (NaN-aware)']
for name in compare_models:
    other_k10 = all_model_results[name][k_ref]
    result = delong_test(
        bnb_k10['y_true'],
        other_k10['y_prob'],
        bnb_k10['y_prob'],
    )
    sig = "Yes" if result['p_value'] < 0.05 else "No"
    print(
        f"  {name:<25} {result['auc_a']:<10.4f} {result['auc_b']:<10.4f} "
        f"{result['diff']:>+10.4f} {result['z_stat']:>10.4f} {result['p_value']:<12.4f} {sig}"
    )

print(f"\n  Note: p < 0.05 indicates statistically significant difference in AUC.")
print(f"  'No' in Significant? column means BNB is not significantly different from that model.")

# ─────────────────────────────────────────────
# 6. McNEMAR TEST — pairwise error-pattern comparison vs BNB (k=10)
# ─────────────────────────────────────────────
print(f"\n{'='*80}")
print("  McNemar Test: Pairwise prediction comparison vs Bernoulli Naive Bayes (k=10)")
print(f"{'='*80}")
print(f"  {'Model':<25} {'b (A+B-)':<10} {'c (A-B+)':<10} {'Statistic':<12} {'p-value':<12} {'Method':<8} {'Significant?'}")
print(f"  {'-'*87}")

for name in compare_models:
    other_k10 = all_model_results[name][k_ref]
    result = mcnemar_test(
        bnb_k10['y_true'],
        other_k10['y_pred'],
        bnb_k10['y_pred'],
    )
    sig = "Yes" if result['p_value'] < 0.05 else "No"
    stat_str = f"{result['statistic']:.4f}" if not np.isnan(result['statistic']) else "—"
    print(
        f"  {name:<25} {result['b']:<10} {result['c']:<10} "
        f"{stat_str:<12} {result['p_value']:<12.4f} {result['method']:<8} {sig}"
    )

print(f"\n  b = instances where the other model is correct but BNB is wrong")
print(f"  c = instances where BNB is correct but the other model is wrong")
print(f"  H0: both classifiers make errors on the same instances (b = c)")
print(f"  p < 0.05 -> significantly different error patterns")

# ─────────────────────────────────────────────
# 7. PAIRED BOOTSTRAP DIFFERENCE CIs - BNB minus comparator (k=10)
# ─────────────────────────────────────────────
print(f"\n{'='*104}")
print("  Paired Bootstrap 95% CIs: Bernoulli Naive Bayes minus comparator (k=10)")
print(f"{'='*104}")
print(f"  {'Model':<25} {'Acc diff [95% CI]':<24} {'F1 diff [95% CI]':<24} {'AUC diff [95% CI]':<24}")
print(f"  {'-'*101}")

for name in compare_models:
    other_k10 = all_model_results[name][k_ref]
    diff_result = paired_bootstrap_diff_ci(
        bnb_k10['y_true'],
        bnb_k10['y_pred'],
        bnb_k10['y_prob'],
        other_k10['y_pred'],
        other_k10['y_prob'],
    )

    def _fmt_diff(metric):
        diff = diff_result['observed_diff'][metric]
        lo, hi = diff_result['ci'][metric]
        return f"{diff:+.3f} [{lo:+.3f}, {hi:+.3f}]"

    print(
        f"  {name:<25} "
        f"{_fmt_diff('accuracy'):<24} "
        f"{_fmt_diff('f1'):<24} "
        f"{_fmt_diff('auc'):<24}"
    )

print(f"\n  Positive differences favor Bernoulli Naive Bayes.")
print(f"  If a 95% CI includes 0, the paired bootstrap does not show a clear difference on that metric.")


# ─────────────────────────────────────────────
# 8. DIRECT: BNB (NaN-aware) vs BNB (mean imputed) — k=10
# ─────────────────────────────────────────────
bnb_nan_k10 = bnb_nan_results[k_ref]

delong_direct = delong_test(bnb_nan_k10['y_true'], bnb_nan_k10['y_prob'], bnb_k10['y_prob'])
mcnem_direct  = mcnemar_test(bnb_nan_k10['y_true'], bnb_nan_k10['y_pred'], bnb_k10['y_pred'])
boot_direct   = paired_bootstrap_diff_ci(
    bnb_nan_k10['y_true'],
    bnb_nan_k10['y_pred'], bnb_nan_k10['y_prob'],
    bnb_k10['y_pred'],     bnb_k10['y_prob'],
)

print(f"\n{'='*80}")
print(f"  Direct comparison: BNB (NaN-aware) vs BNB (mean imputed) — k={k_ref}")
print(f"{'='*80}")
print(f"  DeLong:  AUC_nan={delong_direct['auc_a']:.4f}  AUC_imp={delong_direct['auc_b']:.4f}"
      f"  diff={delong_direct['diff']:+.4f}  Z={delong_direct['z_stat']:.4f}"
      f"  p={delong_direct['p_value']:.4f}  sig={'Yes' if delong_direct['p_value'] < 0.05 else 'No'}")
stat_str = f"{mcnem_direct['statistic']:.4f}" if not np.isnan(mcnem_direct['statistic']) else "—"
print(f"  McNemar: b={mcnem_direct['b']}  c={mcnem_direct['c']}"
      f"  stat={stat_str}  p={mcnem_direct['p_value']:.4f}  method={mcnem_direct['method']}"
      f"  sig={'Yes' if mcnem_direct['p_value'] < 0.05 else 'No'}")

def _fmt_direct(metric):
    diff = boot_direct['observed_diff'][metric]
    lo, hi = boot_direct['ci'][metric]
    return f"{diff:+.3f} [{lo:+.3f}, {hi:+.3f}]"

print(f"  Bootstrap (NaN-aware minus imputed):")
print(f"    Acc  {_fmt_direct('accuracy')}")
print(f"    F1   {_fmt_direct('f1')}")
print(f"    AUC  {_fmt_direct('auc')}")
print(f"  Positive values mean BNB (NaN-aware) outperforms BNB (mean imputed).")


# ─────────────────────────────────────────────
# 9. DELONG TEST — pairwise AUC comparison vs BNB_nan (k=10)
# ─────────────────────────────────────────────
print(f"\n{'='*80}")
print("  DeLong Test: Pairwise AUC comparison vs BNB (NaN-aware) (k=10)")
print(f"{'='*80}")
print(f"  {'Model':<25} {'AUC_model':<10} {'AUC_BNBnan':<10} {'Diff':<10} {'Z':<10} {'p-value':<12} {'Significant?'}")
print(f"  {'-'*77}")

compare_models_nan = list(models.keys()) + ['Bernoulli Naive Bayes']
for name in compare_models_nan:
    other_k10 = all_model_results[name][k_ref]
    result = delong_test(
        bnb_nan_k10['y_true'],
        other_k10['y_prob'],
        bnb_nan_k10['y_prob'],
    )
    sig = "Yes" if result['p_value'] < 0.05 else "No"
    print(
        f"  {name:<25} {result['auc_a']:<10.4f} {result['auc_b']:<10.4f} "
        f"{result['diff']:>+10.4f} {result['z_stat']:>10.4f} {result['p_value']:<12.4f} {sig}"
    )

print(f"\n  Note: p < 0.05 indicates statistically significant difference in AUC.")
print(f"  'No' means BNB (NaN-aware) is not significantly different from that model.")


# ─────────────────────────────────────────────
# 10. McNEMAR TEST — pairwise error-pattern comparison vs BNB_nan (k=10)
# ─────────────────────────────────────────────
print(f"\n{'='*80}")
print("  McNemar Test: Pairwise prediction comparison vs BNB (NaN-aware) (k=10)")
print(f"{'='*80}")
print(f"  {'Model':<25} {'b (A+B-)':<10} {'c (A-B+)':<10} {'Statistic':<12} {'p-value':<12} {'Method':<8} {'Significant?'}")
print(f"  {'-'*87}")

for name in compare_models_nan:
    other_k10 = all_model_results[name][k_ref]
    result = mcnemar_test(
        bnb_nan_k10['y_true'],
        other_k10['y_pred'],
        bnb_nan_k10['y_pred'],
    )
    sig = "Yes" if result['p_value'] < 0.05 else "No"
    stat_str = f"{result['statistic']:.4f}" if not np.isnan(result['statistic']) else "—"
    print(
        f"  {name:<25} {result['b']:<10} {result['c']:<10} "
        f"{stat_str:<12} {result['p_value']:<12.4f} {result['method']:<8} {sig}"
    )

print(f"\n  b = instances where the other model is correct but BNB (NaN-aware) is wrong")
print(f"  c = instances where BNB (NaN-aware) is correct but the other model is wrong")
print(f"  H0: both classifiers make errors on the same instances (b = c)")
print(f"  p < 0.05 -> significantly different error patterns")


# ─────────────────────────────────────────────
# 11. PAIRED BOOTSTRAP DIFFERENCE CIs — BNB_nan minus comparator (k=10)
# ─────────────────────────────────────────────
print(f"\n{'='*104}")
print("  Paired Bootstrap 95% CIs: BNB (NaN-aware) minus comparator (k=10)")
print(f"{'='*104}")
print(f"  {'Model':<25} {'Acc diff [95% CI]':<24} {'F1 diff [95% CI]':<24} {'AUC diff [95% CI]':<24}")
print(f"  {'-'*101}")

for name in compare_models_nan:
    other_k10 = all_model_results[name][k_ref]
    diff_result = paired_bootstrap_diff_ci(
        bnb_nan_k10['y_true'],
        bnb_nan_k10['y_pred'],
        bnb_nan_k10['y_prob'],
        other_k10['y_pred'],
        other_k10['y_prob'],
    )

    def _fmt_nan_diff(metric):
        diff = diff_result['observed_diff'][metric]
        lo, hi = diff_result['ci'][metric]
        return f"{diff:+.3f} [{lo:+.3f}, {hi:+.3f}]"

    print(
        f"  {name:<25} "
        f"{_fmt_nan_diff('accuracy'):<24} "
        f"{_fmt_nan_diff('f1'):<24} "
        f"{_fmt_nan_diff('auc'):<24}"
    )

print(f"\n  Positive differences favor BNB (NaN-aware).")
print(f"  If a 95% CI includes 0, the paired bootstrap does not show a clear difference on that metric.")

print("\n=== Done ===")
