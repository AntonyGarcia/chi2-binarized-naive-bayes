import io
import json
import os
import re
import sys
import unicodedata
import warnings
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

# Avoid a known Windows loky/core-count noise line in some sklearn/joblib setups.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

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

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
SEED = 42
K_VALUES = [5, 10]

RF_TREES = 500
ET_TREES = 500
MLP_HIDDEN = (64, 32)
MLP_MAX_ITER = 2000
CAT_ITERS = 300
STACK_CV = 5

MODEL_LABELS = {
    "LogisticRegression": "Logistic Regression",
    "RandomForest": "Random Forest",
    "ExtraTrees": "Extra Trees",
    "MLP": "Multilayer Perceptron (MLP)",
    "CatBoost": "CatBoost" if HAS_CATBOOST else "CatBoost-style GBDT Proxy",
    "Stacking": "Stacking Ensemble (Liu et al.)",
    "Bernoulli Naive Bayes": "Bernoulli Naive Bayes (BNB)",
}

MODEL_ORDER = [
    "LogisticRegression",
    "RandomForest",
    "ExtraTrees",
    "MLP",
    "CatBoost",
    "Stacking",
]


def safe_div(num, den):
    return float(num / den) if den != 0 else float("nan")


def _canon(name):
    folded = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", folded.lower())


def resolve_columns(requested_names, available_cols):
    if not requested_names:
        return []
    canon_map = {_canon(c): c for c in available_cols}
    resolved = []
    for name in requested_names:
        key = _canon(name)
        if key in canon_map:
            resolved.append(canon_map[key])
    dedup = []
    seen = set()
    for c in resolved:
        if c not in seen:
            dedup.append(c)
            seen.add(c)
    return dedup


def load_metadata(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(csv_path, metadata):
    df = pd.read_csv(csv_path)
    original_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)

    target_col = resolve_columns(["HeartDisease"], df.columns)
    if not target_col:
        raise ValueError("Could not resolve target column (expected HeartDisease).")
    target_col = target_col[0]

    removed_feature_meta = (
        metadata.get("dataset", {})
        .get("primary_dataset", {})
        .get("feature_removed", "RestingECG")
    )
    removed_feature = None

    X_df = df.drop(columns=[target_col]).copy()
    y = pd.to_numeric(df[target_col], errors="coerce")

    drop_candidates = resolve_columns([removed_feature_meta, "RestingECG"], X_df.columns)
    if drop_candidates:
        removed_feature = drop_candidates[0]
        X_df = X_df.drop(columns=[removed_feature])

    encoded_cols = []
    for col in X_df.columns:
        if not is_numeric_dtype(X_df[col]):
            X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))
            encoded_cols.append(col)

    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    valid_mask = (~X_df.isna().any(axis=1)) & (~y.isna())
    X_df = X_df.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True).astype(int)

    if set(np.unique(y)) != {0, 1}:
        uniq = sorted(pd.unique(y))
        raise ValueError(f"Target must be binary 0/1. Got: {uniq}")

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_df.values.astype(float))
    y = y.values

    summary = {
        "feature_names": list(X_df.columns),
        "raw_rows": int(original_rows),
        "dedup_rows": int(len(df)),
        "clean_rows": int(len(X)),
        "removed_rows_after_cleaning": int(len(df) - len(X)),
        "n_features": int(X.shape[1]),
        "target_col": target_col,
        "removed_feature": removed_feature,
        "encoded_cols": encoded_cols,
        "class_counts": {
            "negative_0": int(np.sum(y == 0)),
            "positive_1": int(np.sum(y == 1)),
        },
    }
    return X, y, summary


def _catboost_or_proxy(seed):
    if HAS_CATBOOST:
        return CatBoostClassifier(
            iterations=CAT_ITERS,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )

    # Fallback approximation when catboost is unavailable.
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=CAT_ITERS,
        random_state=seed,
    )


def _stacking_estimators(seed):
    return [
        ("lr", LogisticRegression(solver="lbfgs", max_iter=3000, random_state=seed)),
        ("rf", RandomForestClassifier(n_estimators=RF_TREES, random_state=seed, n_jobs=-1)),
        ("et", ExtraTreesClassifier(n_estimators=ET_TREES, random_state=seed, n_jobs=-1)),
        (
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=MLP_HIDDEN,
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=MLP_MAX_ITER,
                random_state=seed,
            ),
        ),
        ("cat", _catboost_or_proxy(seed)),
    ]


def build_model(model_key, seed):
    if model_key == "LogisticRegression":
        return LogisticRegression(
            solver="lbfgs",
            max_iter=3000,
            random_state=seed,
        )

    if model_key == "RandomForest":
        return RandomForestClassifier(
            n_estimators=RF_TREES,
            random_state=seed,
            n_jobs=-1,
        )

    if model_key == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=ET_TREES,
            random_state=seed,
            n_jobs=-1,
        )

    if model_key == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=MLP_MAX_ITER,
            random_state=seed,
        )

    if model_key == "CatBoost":
        return _catboost_or_proxy(seed)

    if model_key == "Stacking":
        return StackingClassifier(
            estimators=_stacking_estimators(seed),
            final_estimator=LogisticRegression(solver="lbfgs", max_iter=3000, random_state=seed),
            cv=STACK_CV,
            stack_method="predict_proba",
            n_jobs=-1,
            passthrough=False,
        )

    raise ValueError(f"Unknown model_key: {model_key}")


def compute_classwise_metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    return {
        "negative_0": {"precision": float(p[0]), "sensitivity": float(r[0]), "f1": float(f1[0])},
        "positive_1": {"precision": float(p[1]), "sensitivity": float(r[1]), "f1": float(f1[1])},
    }


def evaluate_model(model_key, X, y, k_values):
    name = MODEL_LABELS[model_key]
    W = 24

    def _fmt(val, ci_tuple):
        return f"{val:.3f} [{ci_tuple[0]:.3f}, {ci_tuple[1]:.3f}]"

    print(f"\n{'=' * 125}")
    print(f"  {name}")
    print(f"{'=' * 125}")
    print(f"  {'k':<5} {'Accuracy':<{W}} {'Precision':<{W}} {'Recall':<{W}} {'F-Score':<{W}} {'AUC':<{W}}")
    print(f"  {'-' * 122}")

    all_results = {}

    for k in k_values:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
        y_true_all, y_pred_all, y_prob_all = [], [], []

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = build_model(model_key, SEED + fold)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

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
        mcc = matthews_corrcoef(y_true_all, y_pred_all)
        tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1]).ravel()
        sensitivity = safe_div(tp, tp + fn)
        specificity = safe_div(tn, tn + fp)
        ci = bootstrap_ci(y_true_all, y_pred_all, y_prob_all)

        print(
            f"  {k:<5} {_fmt(acc, ci['accuracy']):<{W}} {_fmt(prec, ci['precision']):<{W}} "
            f"{_fmt(rec, ci['recall']):<{W}} {_fmt(f1, ci['f1']):<{W}} {_fmt(auc, ci['auc']):<{W}}"
        )

        all_results[k] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
            "mcc": mcc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "ci": ci,
            "y_true": y_true_all,
            "y_pred": y_pred_all,
            "y_prob": y_prob_all,
        }

    k_cm = 10 if 10 in all_results else list(all_results.keys())[-1]
    y_true_cm = all_results[k_cm]["y_true"]
    y_pred_cm = all_results[k_cm]["y_pred"]
    tn, fp, fn, tp = confusion_matrix(y_true_cm, y_pred_cm, labels=[0, 1]).ravel()
    classwise = compute_classwise_metrics(y_true_cm, y_pred_cm)
    c0 = classwise["negative_0"]
    c1 = classwise["positive_1"]

    print(f"\n  Confusion Matrix (k={k_cm}):")
    print(f"    TN={tn}  FP={fp}")
    print(f"    FN={fn}  TP={tp}")
    print(f"  Sensitivity (k={k_cm})   : {all_results[k_cm]['sensitivity']:.3f}")
    print(f"  Specificity (k={k_cm})   : {all_results[k_cm]['specificity']:.3f}")
    print(f"  MCC (k={k_cm})           : {all_results[k_cm]['mcc']:.3f}")
    print(
        "  Classwise Negative       : "
        f"P={c0['precision']:.3f}  S={c0['sensitivity']:.3f}  F1={c0['f1']:.3f}"
    )
    print(
        "  Classwise Positive       : "
        f"P={c1['precision']:.3f}  S={c1['sensitivity']:.3f}  F1={c1['f1']:.3f}"
    )

    return all_results


def print_dataset_summary(summary, metadata):
    fused_note = (
        metadata.get("dataset", {})
        .get("primary_dataset", {})
        .get("name", "Fused Heart Dataset")
    )
    print("=== Liu et al. Heart Disease Pipeline ===")
    print("Consistent cross-validated evaluation + BNB comparison")
    print(f"Dataset note                   : {fused_note}")
    print(f"Rows (raw -> dedup -> clean)   : {summary['raw_rows']} -> {summary['dedup_rows']} -> {summary['clean_rows']}")
    print(f"Removed rows after cleaning    : {summary['removed_rows_after_cleaning']}")
    print(f"Features                       : {summary['n_features']}")
    print(f"Target                         : {summary['target_col']}")
    print(f"Feature removed (SHAP/GBDT)    : {summary['removed_feature']}")
    print(
        f"Class counts (0/1)             : "
        f"{summary['class_counts']['negative_0']} / {summary['class_counts']['positive_1']}"
    )
    print(f"Label encoded columns          : {summary['encoded_cols']}")
    print(f"K values                       : {K_VALUES}")
    print("Paper split note               : 80/20 in paper; k-fold CV used here for consistency")
    if not HAS_CATBOOST:
        print("CatBoost note                  : catboost package not found, using GBDT proxy")


def print_reported_reference(metadata):
    heart_res = metadata.get("results", {}).get("heart_dataset", {})
    stack = heart_res.get("stacking_model", {})

    print("\nReported paper heart-dataset metrics:")
    print("  Model                          Acc(%)  Prec(%)  Rec(%)  F1(%)   AUC")
    print("  ----------------------------------------------------------------------")

    best_single = heart_res.get("best_single_model", "CatBoost")
    best_single_acc = heart_res.get("best_single_accuracy", float("nan"))
    print(f"  {best_single:<30} {best_single_acc:>6.2f}  {'-':>7}  {'-':>6}  {'-':>6}  {'-':>5}")

    if stack:
        print(
            f"  {'Stacking Ensemble':<30} "
            f"{stack.get('accuracy', float('nan')):>6.2f}  "
            f"{stack.get('precision', float('nan')):>7.2f}  "
            f"{stack.get('recall', float('nan')):>6.2f}  "
            f"{stack.get('f1_score', float('nan')):>6.2f}  "
            f"{stack.get('auc', float('nan')):>5.2f}"
        )
        print("  Stacking is reported as best model in Liu et al.")

    gen = metadata.get("results", {}).get("heart_attack_dataset", {}).get("stacking_model", {})
    if gen:
        print(
            "\nReported secondary-dataset (Heart Attack) stacking:"
            f" Acc={gen.get('accuracy', float('nan')):.2f}%,"
            f" Prec={gen.get('precision', float('nan')):.2f}%,"
            f" Rec={gen.get('recall', float('nan')):.2f}%,"
            f" F1={gen.get('f1_score', float('nan')):.2f}%,"
            f" AUC={gen.get('auc', float('nan')):.2f}"
        )


def print_bnb_results(bnb_results, k_values):
    W = 24

    def _fmt(val, ci_tuple):
        return f"{val:.3f} [{ci_tuple[0]:.3f}, {ci_tuple[1]:.3f}]"

    print(f"\n{'=' * 125}")
    print("  Bernoulli Naive Bayes (BNB)")
    print(f"{'=' * 125}")
    print(f"  {'k':<5} {'Accuracy':<{W}} {'Precision':<{W}} {'Recall':<{W}} {'F-Score':<{W}} {'AUC':<{W}}")
    print(f"  {'-' * 122}")

    for k in k_values:
        r = bnb_results[k]
        print(
            f"  {k:<5} {_fmt(r['accuracy'], r['ci']['accuracy']):<{W}} "
            f"{_fmt(r['precision'], r['ci']['precision']):<{W}} "
            f"{_fmt(r['recall'], r['ci']['recall']):<{W}} "
            f"{_fmt(r['f1'], r['ci']['f1']):<{W}} {_fmt(r['auc'], r['ci']['auc']):<{W}}"
        )

    k_cm = 10 if 10 in bnb_results else list(bnb_results.keys())[-1]
    y_true_cm = bnb_results[k_cm]["y_true"]
    y_pred_cm = bnb_results[k_cm]["y_pred"]
    cm = bnb_results[k_cm]["confusion_matrix"]
    tn, fp, fn, tp = cm["TN"], cm["FP"], cm["FN"], cm["TP"]
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    mcc = matthews_corrcoef(y_true_cm, y_pred_cm)
    classwise = compute_classwise_metrics(y_true_cm, y_pred_cm)
    c0 = classwise["negative_0"]
    c1 = classwise["positive_1"]

    print(f"\n  Confusion Matrix (k={k_cm}):")
    print(f"    TN={tn}  FP={fp}")
    print(f"    FN={fn}  TP={tp}")
    print(f"  Sensitivity (k={k_cm})   : {sensitivity:.3f}")
    print(f"  Specificity (k={k_cm})   : {specificity:.3f}")
    print(f"  MCC (k={k_cm})           : {mcc:.3f}")
    print(
        "  Classwise Negative       : "
        f"P={c0['precision']:.3f}  S={c0['sensitivity']:.3f}  F1={c0['f1']:.3f}"
    )
    print(
        "  Classwise Positive       : "
        f"P={c1['precision']:.3f}  S={c1['sensitivity']:.3f}  F1={c1['f1']:.3f}"
    )


def print_bnb_comparisons(all_model_results, bnb_results):
    k_ref = 10
    if k_ref not in bnb_results:
        print(f"\nBNB comparison skipped because k={k_ref} results are unavailable.")
        return

    bnb_k10 = bnb_results[k_ref]

    print(f"\n{'=' * 96}")
    print("  DeLong Test: Pairwise AUC comparison vs Bernoulli Naive Bayes (k=10)")
    print(f"{'=' * 96}")
    print(f"  {'Model':<40} {'AUC_model':<10} {'AUC_BNB':<10} {'Diff':<10} {'Z':<10} {'p-value':<12} {'Significant?'}")
    print(f"  {'-' * 108}")

    for key in MODEL_ORDER:
        other_k10 = all_model_results[key][k_ref]
        stat = delong_test(
            bnb_k10["y_true"],
            other_k10["y_prob"],
            bnb_k10["y_prob"],
        )
        sig = "Yes" if stat["p_value"] < 0.05 else "No"
        print(
            f"  {MODEL_LABELS[key]:<40} {stat['auc_a']:<10.3f} {stat['auc_b']:<10.3f} "
            f"{stat['diff']:>+10.3f} {stat['z_stat']:>10.3f} {stat['p_value']:<12.3f} {sig}"
        )

    print(f"\n{'=' * 96}")
    print("  McNemar Test: Pairwise prediction comparison vs Bernoulli Naive Bayes (k=10)")
    print(f"{'=' * 96}")
    print(f"  {'Model':<40} {'b (A+B-)':<10} {'c (A-B+)':<10} {'Statistic':<12} {'p-value':<12} {'Method':<8} {'Significant?'}")
    print(f"  {'-' * 112}")

    for key in MODEL_ORDER:
        other_k10 = all_model_results[key][k_ref]
        stat = mcnemar_test(
            bnb_k10["y_true"],
            other_k10["y_pred"],
            bnb_k10["y_pred"],
        )
        sig = "Yes" if stat["p_value"] < 0.05 else "No"
        stat_str = f"{stat['statistic']:.3f}" if not np.isnan(stat["statistic"]) else "-"
        print(
            f"  {MODEL_LABELS[key]:<40} {stat['b']:<10} {stat['c']:<10} "
            f"{stat_str:<12} {stat['p_value']:<12.3f} {stat['method']:<8} {sig}"
        )

    print(f"\n{'=' * 112}")
    print("  Paired Bootstrap 95% CIs: Bernoulli Naive Bayes minus comparator (k=10)")
    print(f"{'=' * 112}")
    print(f"  {'Model':<40} {'Acc diff [95% CI]':<24} {'F1 diff [95% CI]':<24} {'AUC diff [95% CI]':<24}")
    print(f"  {'-' * 112}")

    for key in MODEL_ORDER:
        other_k10 = all_model_results[key][k_ref]
        diff = paired_bootstrap_diff_ci(
            bnb_k10["y_true"],
            bnb_k10["y_pred"],
            bnb_k10["y_prob"],
            other_k10["y_pred"],
            other_k10["y_prob"],
        )

        def _fmt(metric):
            d = diff["observed_diff"][metric]
            lo, hi = diff["ci"][metric]
            return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"

        print(
            f"  {MODEL_LABELS[key]:<40} "
            f"{_fmt('accuracy'):<24} {_fmt('f1'):<24} {_fmt('auc'):<24}"
        )

    print("\n  Positive differences favor Bernoulli Naive Bayes.")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "..", "..", "datasets", "heart_disease.csv")
    metadata_path = os.path.join(base_dir, "metadata", "liu_et_al.json")

    metadata = load_metadata(metadata_path)
    X, y, summary = load_dataset(dataset_path, metadata)

    print_dataset_summary(summary, metadata)
    print_reported_reference(metadata)

    all_model_results = {}
    for key in MODEL_ORDER:
        all_model_results[key] = evaluate_model(key, X, y, K_VALUES)

    # Suppress default bnb.evaluate printing (2-decimal formatting), then print custom 3-decimal table.
    with redirect_stdout(io.StringIO()):
        bnb_results = bnb.evaluate(
            X, y, k_values=K_VALUES,
            categorical_idx=bnb.categorical_indices(summary["feature_names"]),
        )
    all_model_results["Bernoulli Naive Bayes"] = bnb_results
    print_bnb_results(bnb_results, K_VALUES)

    print_bnb_comparisons(all_model_results, bnb_results)
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
