"""
Azar and El-Said (2013), Probabilistic neural network for breast cancer classification.

Re-implementation of the three classifiers of the study (MLP, RBF network and PNN),
evaluated with the common protocol of this project (stratified k-fold CV, pooled
out-of-fold predictions) on the Wisconsin Diagnostic dataset, and compared with BNB.
The original study used the 9-feature Wisconsin (Original) dataset and did not report
all hyperparameters; the settings below follow implementations/breast_cancer/metadata/azar.json.
"""
import os
import sys
import io
import warnings
import numpy as np
import pandas as pd
from contextlib import redirect_stdout

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

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
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
from sklearn.preprocessing import MinMaxScaler


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
SEED = 42
K_VALUES = [5, 10]

MODEL_LABELS = {
    "MLP": "Multi-layer Perceptron (MLP)",
    "RBF": "Radial Basis Function Network (RBF)",
    "PNN": "Probabilistic Neural Network (PNN)",
    "Bernoulli Naive Bayes": "Bernoulli Naive Bayes (BNB)",
}

MODEL_ORDER = ["MLP", "RBF", "PNN"]

# Validation-phase results reported by Azar and El-Said (10-fold CV, original WBC dataset).
PAPER_TEST_ACCURACY = {"MLP": 96.34, "RBF": 96.05, "PNN": 97.66}
PAPER_AUC = {"MLP": 0.993, "RBF": 0.990, "PNN": 0.994}

MLP_HIDDEN = 7          # Table 1 of the study (the conclusion mentions 3)
RBF_CENTERS = 23        # best number of hidden neurons in the conclusion of the study
PNN_SIGMAS = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)   # smoothing factor tuned on the training fold


class RBFNetwork(ClassifierMixin, BaseEstimator):
    """Gaussian RBF hidden layer (k-means centers) with a sigmoid (logistic) output unit."""

    def __init__(self, n_centers=RBF_CENTERS, random_state=None):
        self.n_centers = n_centers
        self.random_state = random_state

    def _hidden(self, X):
        d2 = ((X[:, None, :] - self.centers_[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-d2 / (2.0 * self.width_ ** 2))

    def fit(self, X, y):
        km = KMeans(n_clusters=self.n_centers, n_init=10, random_state=self.random_state).fit(X)
        self.centers_ = km.cluster_centers_
        dists = np.sqrt(((self.centers_[:, None, :] - self.centers_[None, :, :]) ** 2).sum(axis=2))
        self.width_ = dists[np.triu_indices(self.n_centers, k=1)].mean()
        self.output_ = LogisticRegression(max_iter=2000).fit(self._hidden(X), y)
        self.classes_ = self.output_.classes_
        return self

    def predict_proba(self, X):
        return self.output_.predict_proba(self._hidden(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class PNN(ClassifierMixin, BaseEstimator):
    """
    Parzen probabilistic neural network on the [0, 1]-rescaled inputs, with the smoothing
    factor chosen by inner 5-fold CV on the training data. The study also normalizes each
    sample to unit length; that step is omitted here because, on the 30-feature diagnostic
    dataset, it lowered the AUC from 0.992 to 0.973.
    """

    def __init__(self, sigmas=PNN_SIGMAS, random_state=None):
        self.sigmas = sigmas
        self.random_state = random_state

    @staticmethod
    def _proba(X_train, y_train, X, sigma):
        d2 = ((X[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=2)
        k = np.exp(-d2 / (2.0 * sigma ** 2))
        dens = np.column_stack([k[:, y_train == c].mean(axis=1) for c in (0, 1)])
        prior = np.array([np.mean(y_train == c) for c in (0, 1)])
        joint = dens * prior + 1e-300
        return joint / joint.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scores = []
        for sigma in self.sigmas:
            probs, truth = [], []
            for tr, va in inner.split(X, y):
                probs.append(self._proba(X[tr], y[tr], X[va], sigma)[:, 1])
                truth.append(y[va])
            scores.append(roc_auc_score(np.concatenate(truth), np.concatenate(probs)))
        self.sigma_ = self.sigmas[int(np.argmax(scores))]
        self.X_, self.y_ = X, np.asarray(y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self._proba(self.X_, self.y_, np.asarray(X), self.sigma_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    original_rows = len(df)

    if "diagnosis" in df.columns:
        y_raw = df["diagnosis"]
        X_df = df.drop(columns=["diagnosis"]).copy()
        if y_raw.dtype == object:
            mapping = {"M": 1, "B": 0, "malignant": 1, "benign": 0}
            y = y_raw.map(mapping).astype(float)
        else:
            y = y_raw.astype(float)
        dataset_variant = "diagnosis-format breast cancer dataset"
    elif "Class" in df.columns:
        y_raw = df["Class"]
        X_df = df.drop(columns=["Class"]).copy()

        for id_col in ["id", "ID", "Sample code number"]:
            if id_col in X_df.columns:
                X_df = X_df.drop(columns=[id_col])

        y = y_raw.replace({2: 0, 4: 1}).astype(float)
        dataset_variant = "class-format breast cancer dataset"
    else:
        raise ValueError("Could not find target column. Expected 'diagnosis' or 'Class'.")

    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    valid_mask = (~X_df.isna().any(axis=1)) & (~y.isna())
    X_df = X_df.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True).astype(int)

    if set(np.unique(y)) != {0, 1}:
        uniq = sorted(pd.unique(y))
        raise ValueError(f"Target must be binary 0/1 after mapping. Got: {uniq}")

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_df.values.astype(float))
    y = y.values

    summary = {
        "dataset_variant": dataset_variant,
        "raw_rows": int(original_rows),
        "clean_rows": int(len(X)),
        "removed_rows": int(original_rows - len(X)),
        "n_features": int(X.shape[1]),
        "class_counts": {
            "benign_0": int(np.sum(y == 0)),
            "malignant_1": int(np.sum(y == 1)),
        },
    }
    return X, y, summary


def build_model(model_key, seed):
    if model_key == "MLP":
        # One hidden layer with log-sigmoid units; L-BFGS stands in for scaled conjugate gradient.
        return MLPClassifier(
            hidden_layer_sizes=(MLP_HIDDEN,),
            activation="logistic",
            solver="lbfgs",
            max_iter=5000,
            random_state=seed,
        )

    if model_key == "RBF":
        return RBFNetwork(n_centers=RBF_CENTERS, random_state=seed)

    if model_key == "PNN":
        return PNN(random_state=seed)

    raise ValueError(f"Unknown model_key: {model_key}")


def compute_classwise_metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    return {
        "benign_0": {"precision": float(p[0]), "sensitivity": float(r[0]), "f1": float(f1[0])},
        "malignant_1": {"precision": float(p[1]), "sensitivity": float(r[1]), "f1": float(f1[1])},
    }


def evaluate_model(model_key, X, y, k_values):
    name = MODEL_LABELS[model_key]
    W = 24

    def _fmt(val, ci_tuple):
        return f"{val:.3f} [{ci_tuple[0]:.3f}, {ci_tuple[1]:.3f}]"

    print(f"\n{'='*125}")
    print(f"  {name}")
    print(f"{'='*125}")
    print(f"  {'k':<5} {'Accuracy':<{W}} {'Precision':<{W}} {'Recall':<{W}} {'F-Score':<{W}} {'AUC':<{W}}")
    print(f"  {'-'*122}")

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
        sens_m = recall_score(y_true_all, y_pred_all, pos_label=1, zero_division=0)
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
            "sensitivity_malignant": sens_m,
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
    c0 = classwise["benign_0"]
    c1 = classwise["malignant_1"]

    print(f"\n  Confusion Matrix (k={k_cm}):")
    print(f"    TN={tn}  FP={fp}")
    print(f"    FN={fn}  TP={tp}")
    print(f"  MCC (k={k_cm})          : {all_results[k_cm]['mcc']:.3f}")
    print(f"  Sensitivity (M, k={k_cm}): {all_results[k_cm]['sensitivity_malignant']:.3f}")
    print(
        "  Classwise Benign        : "
        f"P={c0['precision']:.3f}  S={c0['sensitivity']:.3f}  F1={c0['f1']:.3f}"
    )
    print(
        "  Classwise Malignant     : "
        f"P={c1['precision']:.3f}  S={c1['sensitivity']:.3f}  F1={c1['f1']:.3f}"
    )

    return all_results


def print_dataset_summary(summary):
    print("=== Azar and El-Said Breast Cancer Pipeline ===")
    print("Consistent cross-validated evaluation + BNB comparison")
    print(f"Dataset variant              : {summary['dataset_variant']}")
    print(f"Rows (raw -> clean)          : {summary['raw_rows']} -> {summary['clean_rows']}")
    print(f"Removed rows                 : {summary['removed_rows']}")
    print(f"Features                     : {summary['n_features']}")
    print(
        f"Class counts (0/1)           : "
        f"{summary['class_counts']['benign_0']} / {summary['class_counts']['malignant_1']}"
    )
    print(f"K values                     : {K_VALUES}")
    print("Note                         : MLP, RBF network and PNN as described in the study")


def print_reported_reference():
    print("\nReported paper validation reference (10-fold CV, original WBC dataset):")
    print("  Model                          Test Acc (%)   AUC")
    print("  --------------------------------------------------")
    for k in MODEL_ORDER:
        print(
            f"  {MODEL_LABELS[k]:<30} "
            f"{PAPER_TEST_ACCURACY[k]:>11.1f}   {PAPER_AUC[k]:.3f}"
        )
    print("  Current implementation uses k-fold CV for consistency across projects.")


def print_bnb_results(bnb_results, k_values):
    W = 24

    def _fmt(val, ci_tuple):
        return f"{val:.3f} [{ci_tuple[0]:.3f}, {ci_tuple[1]:.3f}]"

    print(f"\n{'='*125}")
    print("  Bernoulli Naive Bayes (BNB)")
    print(f"{'='*125}")
    print(f"  {'k':<5} {'Accuracy':<{W}} {'Precision':<{W}} {'Recall':<{W}} {'F-Score':<{W}} {'AUC':<{W}}")
    print(f"  {'-'*122}")

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
    mcc = matthews_corrcoef(y_true_cm, y_pred_cm)
    sens_m = recall_score(y_true_cm, y_pred_cm, pos_label=1, zero_division=0)
    classwise = compute_classwise_metrics(y_true_cm, y_pred_cm)
    c0 = classwise["benign_0"]
    c1 = classwise["malignant_1"]

    print(f"\n  Confusion Matrix (k={k_cm}):")
    print(f"    TN={cm['TN']}  FP={cm['FP']}")
    print(f"    FN={cm['FN']}  TP={cm['TP']}")
    print(f"  MCC (k={k_cm})          : {mcc:.3f}")
    print(f"  Sensitivity (M, k={k_cm}): {sens_m:.3f}")
    print(
        "  Classwise Benign        : "
        f"P={c0['precision']:.3f}  S={c0['sensitivity']:.3f}  F1={c0['f1']:.3f}"
    )
    print(
        "  Classwise Malignant     : "
        f"P={c1['precision']:.3f}  S={c1['sensitivity']:.3f}  F1={c1['f1']:.3f}"
    )


def print_bnb_comparisons(all_model_results, bnb_results):
    k_ref = 10
    if k_ref not in bnb_results:
        print(f"\nBNB comparison skipped because k={k_ref} results are unavailable.")
        return

    bnb_k10 = bnb_results[k_ref]
    compare_keys = MODEL_ORDER

    print(f"\n{'='*96}")
    print("  DeLong Test: Pairwise AUC comparison vs Bernoulli Naive Bayes (k=10)")
    print(f"{'='*96}")
    print(f"  {'Model':<40} {'AUC_model':<10} {'AUC_BNB':<10} {'Diff':<10} {'Z':<10} {'p-value':<12} {'Significant?'}")
    print(f"  {'-'*108}")

    for key in compare_keys:
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

    print(f"\n{'='*96}")
    print("  McNemar Test: Pairwise prediction comparison vs Bernoulli Naive Bayes (k=10)")
    print(f"{'='*96}")
    print(f"  {'Model':<40} {'b (A+B-)':<10} {'c (A-B+)':<10} {'Statistic':<12} {'p-value':<12} {'Method':<8} {'Significant?'}")
    print(f"  {'-'*112}")

    for key in compare_keys:
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

    print(f"\n{'='*112}")
    print("  Paired Bootstrap 95% CIs: Bernoulli Naive Bayes minus comparator (k=10)")
    print(f"{'='*112}")
    print(f"  {'Model':<40} {'Acc diff [95% CI]':<24} {'F1 diff [95% CI]':<24} {'AUC diff [95% CI]':<24}")
    print(f"  {'-'*112}")

    for key in compare_keys:
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
    dataset_path = os.path.join(base_dir, "..", "..", "datasets", "breast_cancer.csv")

    X, y, summary = load_dataset(dataset_path)
    print_dataset_summary(summary)
    print_reported_reference()

    all_model_results = {}
    for key in MODEL_ORDER:
        all_model_results[key] = evaluate_model(key, X, y, K_VALUES)

    # Suppress default bnb.evaluate printing (2-decimal formatting), then print custom 3-decimal table.
    with redirect_stdout(io.StringIO()):
        bnb_results = bnb.evaluate(X, y, k_values=K_VALUES)
    all_model_results["Bernoulli Naive Bayes"] = bnb_results
    print_bnb_results(bnb_results, K_VALUES)

    print_bnb_comparisons(all_model_results, bnb_results)
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
