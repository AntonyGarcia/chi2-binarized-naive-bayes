import os
import sys
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

_BASE = os.path.dirname(os.path.abspath(__file__))
_ALGO_CANDIDATES = [
    os.path.join(_BASE, "..", "..", "algorithms"),
    os.path.join(_BASE, "..", "algorithms"),
]
for _p in _ALGO_CANDIDATES:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import bnb
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
# 2. PREPROCESSING  (mirrors the paper)
#    • Zero values in clinical columns treated as missing → drop those rows
#    • No oversampling / undersampling (paper used the cleaned set for final experiments)
# ─────────────────────────────────────────────
ZERO_AS_MISSING = ['Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI']
df_clean = df.copy()
df_clean[ZERO_AS_MISSING] = df_clean[ZERO_AS_MISSING].replace(0, np.nan)
df_clean.dropna(inplace=True)
df_clean.reset_index(drop=True, inplace=True)

print(f"\n=== After removing missing-value rows ===")
print(f"Instances retained : {len(df_clean)}  (paper reports 392)")
print(df_clean['Outcome'].value_counts().rename({0: 'Negative', 1: 'Positive'}))

X = df_clean.drop(columns='Outcome').values
y = df_clean['Outcome'].values

# ─────────────────────────────────────────────
# 3. MODELS  (hyperparams from Table 5 of the paper)
# ─────────────────────────────────────────────

# J48: find best ccp_alpha via 10-fold CV to approximate WEKA's C=0.25 pessimistic pruning.
# C4.5/J48 uses information gain (entropy), and WEKA's default minNumObj=2.
_cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
_path = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=2, random_state=42) \
            .cost_complexity_pruning_path(X, y)
_best_alpha, _best_score = 0.0, 0.0
for _a in _path.ccp_alphas[:-1]:
    _s = cross_validate(
        DecisionTreeClassifier(criterion='entropy', min_samples_leaf=2, ccp_alpha=_a, random_state=42),
        X, y, cv=_cv10, scoring='accuracy'
    )['test_score'].mean()
    if _s > _best_score:
        _best_score, _best_alpha = _s, _a
print(f"\nJ48 tuned ccp_alpha = {_best_alpha:.6f}  (CV accuracy = {_best_score:.4f})")

models = {
    "Naive Bayes": GaussianNB(),                                    # no tuned HPs reported
    "J48 (Decision Tree)": DecisionTreeClassifier(
        criterion='entropy',                                        # C4.5/J48 uses information gain
        min_samples_leaf=2,                                         # WEKA J48 default: minNumObj=2
        ccp_alpha=_best_alpha,                                      # data-driven ≈ WEKA C=0.25 pruning
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,                                           # paper: 100 trees
        max_features=4,                                             # paper: 4 features per tree
        random_state=42
    ),
    "Logistic Regression": LogisticRegression(
        C=1 / 1e-8,                                                 # paper: ridge R = 1e-8  → C = 1/R
        max_iter=1000,
        random_state=42
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
    W = 24   # column width per metric
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

        acc  = accuracy_score(y_true_all, y_pred_all)
        prec = precision_score(y_true_all, y_pred_all, average='weighted')
        rec  = recall_score(y_true_all, y_pred_all, average='weighted')
        f1   = f1_score(y_true_all, y_pred_all, average='weighted')
        auc  = roc_auc_score(y_true_all, y_prob_all)

        ci = bootstrap_ci(y_true_all, y_pred_all, y_prob_all)

        print(f"  {k:<5} {_fmt(acc,  ci['accuracy']):<{W}} {_fmt(prec, ci['precision']):<{W}}"
              f" {_fmt(rec,  ci['recall']):<{W}} {_fmt(f1,   ci['f1']):<{W}}"
              f" {_fmt(auc,  ci['auc']):<{W}}")

        all_results[k] = {
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc,
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

# BNB - binarization is handled inside bnb.py (per-feature chi-square threshold per training fold)
bnb_results = bnb.evaluate(X, y, k_values=K_VALUES)
all_model_results['Bernoulli Naive Bayes'] = bnb_results

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
for name in models.keys():
    other_k10 = all_model_results[name][k_ref]
    # Both use the same y_true (from the same CV splits)
    result = delong_test(
        bnb_k10['y_true'],
        other_k10['y_prob'],   # Model A
        bnb_k10['y_prob'],     # Model B = BNB
    )
    sig = "Yes" if result['p_value'] < 0.05 else "No"
    print(f"  {name:<25} {result['auc_a']:<10.4f} {result['auc_b']:<10.4f} "
          f"{result['diff']:>+10.4f} {result['z_stat']:>10.4f} {result['p_value']:<12.4f} {sig}")

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

for name in models.keys():
    other_k10 = all_model_results[name][k_ref]
    result = mcnemar_test(
        bnb_k10['y_true'],
        other_k10['y_pred'],   # Model A
        bnb_k10['y_pred'],     # Model B = BNB
    )
    sig = "Yes" if result['p_value'] < 0.05 else "No"
    stat_str = f"{result['statistic']:.4f}" if not np.isnan(result['statistic']) else "—"
    print(f"  {name:<25} {result['b']:<10} {result['c']:<10} {stat_str:<12} {result['p_value']:<12.4f} {result['method']:<8} {sig}")

print(f"\n  b = instances where the other model is correct but BNB is wrong")
print(f"  c = instances where BNB is correct but the other model is wrong")
print(f"  H0: both classifiers make errors on the same instances (b = c)")
print(f"  p < 0.05 -> significantly different error patterns")

# -----------------------------------------------------------------------------
# 7. PAIRED BOOTSTRAP DIFFERENCE CIs - BNB minus comparator (k=10)
# -----------------------------------------------------------------------------
print(f"\n{'='*104}")
print("  Paired Bootstrap 95% CIs: Bernoulli Naive Bayes minus comparator (k=10)")
print(f"{'='*104}")
print(f"  {'Model':<25} {'Acc diff [95% CI]':<24} {'F1 diff [95% CI]':<24} {'AUC diff [95% CI]':<24}")
print(f"  {'-'*101}")

for name in models.keys():
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

print("\n=== Done ===")
