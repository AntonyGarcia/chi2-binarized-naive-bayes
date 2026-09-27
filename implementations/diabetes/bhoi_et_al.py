import os
import sys
import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# Paper algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

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
# 2. PREPROCESSING
#    • Normalize column names
#    • For Pima, treat 0 as missing in:
#      Glucose, BloodPressure, SkinThickness, Insulin, BMI
#    • Remove rows with missing values
# ─────────────────────────────────────────────
COLUMN_RENAME_MAP = {
    "Blood Pressure": "BloodPressure",
    "Skin Thickness": "SkinThickness",
    "Diabetes Pedigree Function": "DiabetesPedigreeFunction",
}

df = df.rename(columns=COLUMN_RENAME_MAP)

EXPECTED_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET = "Outcome"

missing_cols = [c for c in EXPECTED_FEATURES + [TARGET] if c not in df.columns]
if missing_cols:
    raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

df_clean = df[EXPECTED_FEATURES + [TARGET]].copy()

ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

df_clean[ZERO_AS_MISSING] = df_clean[ZERO_AS_MISSING].replace(0, np.nan)
df_clean.dropna(inplace=True)
df_clean.reset_index(drop=True, inplace=True)

print("\n=== After removing rows with disguised missing values ===")
print(f"Instances retained : {len(df_clean)}")
print(df_clean[TARGET].value_counts().rename({0: 'Negative', 1: 'Positive'}))

X = df_clean[EXPECTED_FEATURES].values
y = df_clean[TARGET].values


# ─────────────────────────────────────────────
# 3. MODELS
#    Paper lineup + keep BNB as comparator
# ─────────────────────────────────────────────
models = {
    "CT (Classification Tree)": DecisionTreeClassifier(
        random_state=42
    ),

    "SVM": make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        )
    ),

    "k-NN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=5
        )
    ),

    "NB (GaussianNB)": GaussianNB(),

    "RF (Random Forest)": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=1
    ),

    "NN (MLP)": make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=1000,
            random_state=42
        )
    ),

    "AB (AdaBoost)": AdaBoostClassifier(
        n_estimators=50,
        random_state=42
    ),

    "LR (Logistic Regression)": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
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
    W = 24

    def _fmt(val, ci_tuple):
        return f"{val:.4f} [{ci_tuple[0]:.4f}, {ci_tuple[1]:.4f}]"

    print(f"\n{'='*125}")
    print(f"  {name}")
    print(f"{'='*125}")
    print(f"  {'k':<5} {'Accuracy':<{W}} {'Precision':<{W}} {'Recall':<{W}} {'F1':<{W}} {'AUC':<{W}}")
    print(f"  {'-'*122}")

    cm_pred = None
    all_results = {}

    for k in k_values:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        y_true_all, y_pred_all, y_prob_all = [], [], []

        for train_idx, test_idx in cv.split(X, y):
            est = clone(model)
            est.fit(X[train_idx], y[train_idx])

            y_true_fold = y[test_idx]
            y_pred_fold = est.predict(X[test_idx])
            y_prob_fold = est.predict_proba(X[test_idx])[:, 1]

            y_true_all.append(y_true_fold)
            y_pred_all.append(y_pred_fold)
            y_prob_all.append(y_prob_fold)

        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        y_prob_all = np.concatenate(y_prob_all)

        acc = accuracy_score(y_true_all, y_pred_all)
        prec = precision_score(y_true_all, y_pred_all, zero_division=0)
        rec = recall_score(y_true_all, y_pred_all, zero_division=0)
        f1 = f1_score(y_true_all, y_pred_all, zero_division=0)
        auc = roc_auc_score(y_true_all, y_prob_all)

        ci = bootstrap_ci(y_true_all, y_pred_all, y_prob_all)

        print(
            f"  {k:<5} "
            f"{_fmt(acc,  ci['accuracy']):<{W}} "
            f"{_fmt(prec, ci['precision']):<{W}} "
            f"{_fmt(rec,  ci['recall']):<{W}} "
            f"{_fmt(f1,   ci['f1']):<{W}} "
            f"{_fmt(auc,  ci['auc']):<{W}}"
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

    if cm_pred is not None:
        y_true_cm, y_pred_cm = cm_pred
        tn, fp, fn, tp = confusion_matrix(y_true_cm, y_pred_cm).ravel()
        print(f"\n  Confusion Matrix (10-fold):")
        print(f"    TN={tn}  FP={fp}")
        print(f"    FN={fn}  TP={tp}")

    return all_results


# Collect results from all paper models
all_model_results = {}
for name, model in models.items():
    all_model_results[name] = evaluate_model(name, model, X, y, K_VALUES)

# BNB comparator
bnb_results = bnb.evaluate(X, y, k_values=K_VALUES)
all_model_results['BNB'] = bnb_results


# ─────────────────────────────────────────────
# 5. DELONG TEST — pairwise AUC comparison vs BNB (k=10)
# ─────────────────────────────────────────────
print(f"\n{'='*88}")
print("  DeLong Test: Pairwise AUC comparison vs BNB (k=10)")
print(f"{'='*88}")
print(f"  {'Model':<28} {'AUC_model':<10} {'AUC_BNB':<10} {'Diff':<10} {'Z':<10} {'p-value':<12} {'Significant?'}")
print(f"  {'-'*85}")

k_ref = 10
bnb_k10 = bnb_results[k_ref]

for name in models.keys():
    other_k10 = all_model_results[name][k_ref]

    result = delong_test(
        bnb_k10['y_true'],
        other_k10['y_prob'],
        bnb_k10['y_prob'],
    )

    sig = "Yes" if result['p_value'] < 0.05 else "No"
    print(
        f"  {name:<28} "
        f"{result['auc_a']:<10.4f} "
        f"{result['auc_b']:<10.4f} "
        f"{result['diff']:>+10.4f} "
        f"{result['z_stat']:>10.4f} "
        f"{result['p_value']:<12.4f} "
        f"{sig}"
    )

print(f"\n  Note: p < 0.05 indicates statistically significant difference in AUC.")


# ─────────────────────────────────────────────
# 6. McNEMAR TEST — pairwise error-pattern comparison vs BNB (k=10)
# ─────────────────────────────────────────────
print(f"\n{'='*92}")
print("  McNemar Test: Pairwise prediction comparison vs BNB (k=10)")
print(f"{'='*92}")
print(f"  {'Model':<28} {'b (A+B-)':<10} {'c (A-B+)':<10} {'Statistic':<12} {'p-value':<12} {'Method':<8} {'Significant?'}")
print(f"  {'-'*89}")

for name in models.keys():
    other_k10 = all_model_results[name][k_ref]

    result = mcnemar_test(
        bnb_k10['y_true'],
        other_k10['y_pred'],
        bnb_k10['y_pred'],
    )

    sig = "Yes" if result['p_value'] < 0.05 else "No"
    stat_str = f"{result['statistic']:.4f}" if not np.isnan(result['statistic']) else "—"

    print(
        f"  {name:<28} "
        f"{result['b']:<10} "
        f"{result['c']:<10} "
        f"{stat_str:<12} "
        f"{result['p_value']:<12.4f} "
        f"{result['method']:<8} "
        f"{sig}"
    )

print(f"\n  b = instances where the comparator is correct but BNB is wrong")
print(f"  c = instances where BNB is correct but the comparator is wrong")
print(f"  H0: both classifiers make errors on the same instances (b = c)")


# ─────────────────────────────────────────────
# 7. PAIRED BOOTSTRAP DIFFERENCE CIs — BNB minus comparator (k=10)
# ─────────────────────────────────────────────
print(f"\n{'='*108}")
print("  Paired Bootstrap 95% CIs: BNB minus comparator (k=10)")
print(f"{'='*108}")
print(f"  {'Model':<28} {'Acc diff [95% CI]':<24} {'F1 diff [95% CI]':<24} {'AUC diff [95% CI]':<24}")
print(f"  {'-'*105}")

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
        f"  {name:<28} "
        f"{_fmt_diff('accuracy'):<24} "
        f"{_fmt_diff('f1'):<24} "
        f"{_fmt_diff('auc'):<24}"
    )

print(f"\n  Positive differences favor BNB.")
print(f"  If a 95% CI includes 0, the paired bootstrap does not show a clear difference on that metric.")

print("\n=== Done ===")
