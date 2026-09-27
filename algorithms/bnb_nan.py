"""
Bernoulli Naive Bayes with native NaN support (BNB_nan).

Instead of imputing missing values, this implementation simply skips
missing features when computing the posterior.  Each row can use a
different subset of features — only the likelihoods of *observed*
(non-NaN) features are multiplied in.

Binarization uses the same chi-square threshold selection as bnb.py,
but NaN cells are excluded from candidate evaluation.

Interface mirrors bnb.py so it can be dropped into the same pipelines.
"""

import numpy as np
from scipy.stats import chi2_contingency
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from bnb import bootstrap_ci


# ─────────────────────────────────────────────────────────────
# Threshold selection (NaN-aware)
# ─────────────────────────────────────────────────────────────

def chi2_threshold(x_col, y):
    """
    Select the best binarization threshold for a single feature column.
    Rows where x_col is NaN are excluded from the search.
    """
    mask = ~np.isnan(x_col)
    x_valid = x_col[mask]
    y_valid = y[mask]

    if len(x_valid) == 0:
        return 0.0

    candidates = np.unique(x_valid)
    best_t, best_p = candidates[0], 1.0
    for t in candidates:
        binary = (x_valid > t).astype(int)
        table = np.zeros((2, 2), dtype=int)
        for b, label in zip(binary, y_valid):
            table[int(b), int(label)] += 1
        if table.min() == 0 and (table[0].sum() == 0 or table[1].sum() == 0):
            continue
        _, p, _, _ = chi2_contingency(table)
        if p < best_p:
            best_p, best_t = p, t
    return best_t


# ─────────────────────────────────────────────────────────────
# NaN-aware Bernoulli Naive Bayes model
# ─────────────────────────────────────────────────────────────

class BernoulliNB_NaN:
    """
    Bernoulli Naive Bayes that natively handles NaN at predict time.

    Training: learns P(x_j=1 | y=c) with Laplace smoothing from the
    binarized training data (NaN rows excluded per feature).

    Prediction: for each sample, only observed (non-NaN) features
    contribute to the log-posterior.  Missing features are simply
    skipped — their likelihood ratio is treated as 1 (log-likelihood = 0).
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X_bin, y):
        """
        Fit from already-binarized data.  NaN entries in X_bin are ignored
        per-feature when estimating P(x_j=1 | y=c).
        """
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X_bin.shape[1]

        # Class priors
        class_counts = np.array([(y == c).sum() for c in self.classes_], dtype=float)
        self.log_prior_ = np.log(class_counts / class_counts.sum())

        # Per-feature, per-class P(x_j=1 | y=c) with Laplace smoothing
        self.log_prob_1_ = np.zeros((n_classes, n_features))
        self.log_prob_0_ = np.zeros((n_classes, n_features))

        for ci, c in enumerate(self.classes_):
            class_mask = (y == c)
            X_class = X_bin[class_mask]
            for j in range(n_features):
                col = X_class[:, j]
                valid = ~np.isnan(col)
                n_valid = valid.sum()
                n_ones = col[valid].sum()
                # Laplace-smoothed probability
                p1 = (n_ones + self.alpha) / (n_valid + 2 * self.alpha)
                self.log_prob_1_[ci, j] = np.log(p1)
                self.log_prob_0_[ci, j] = np.log(1.0 - p1)

        return self

    def predict_log_proba(self, X_bin):
        n_samples = X_bin.shape[0]
        n_classes = len(self.classes_)
        log_proba = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            for ci in range(n_classes):
                log_p = self.log_prior_[ci]
                for j in range(X_bin.shape[1]):
                    val = X_bin[i, j]
                    if np.isnan(val):
                        continue  # skip missing feature
                    if val == 1:
                        log_p += self.log_prob_1_[ci, j]
                    else:
                        log_p += self.log_prob_0_[ci, j]
                log_proba[i, ci] = log_p

        return log_proba

    def predict_proba(self, X_bin):
        log_proba = self.predict_log_proba(X_bin)
        # Normalize via log-sum-exp for numerical stability
        max_log = log_proba.max(axis=1, keepdims=True)
        log_proba_shifted = log_proba - max_log
        proba = np.exp(log_proba_shifted)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X_bin):
        log_proba = self.predict_log_proba(X_bin)
        return self.classes_[np.argmax(log_proba, axis=1)]


# ─────────────────────────────────────────────────────────────
# Core function — one train/test split (NaN-aware)
# ─────────────────────────────────────────────────────────────

def train_and_score(X_train, y_train, X_test, y_test):
    """
    Fit BernoulliNB_NaN on (X_train, y_train) and evaluate on (X_test, y_test).

    NaN values in X_train are excluded when picking thresholds and
    estimating likelihoods.  NaN values in X_test cause that feature's
    likelihood to be skipped for that row.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test  = np.asarray(X_test,  dtype=float)

    # Select per-feature threshold from training fold (NaN-aware)
    thresholds = np.array([chi2_threshold(X_train[:, j], y_train)
                           for j in range(X_train.shape[1])])

    # Binarize, preserving NaN
    X_train_bin = np.where(np.isnan(X_train), np.nan,
                           (X_train > thresholds).astype(float))
    X_test_bin  = np.where(np.isnan(X_test), np.nan,
                           (X_test  > thresholds).astype(float))

    model = BernoulliNB_NaN()
    model.fit(X_train_bin, y_train)

    y_pred = model.predict(X_test_bin)
    y_prob = model.predict_proba(X_test_bin)[:, 1]

    return {
        'accuracy'            : accuracy_score(y_test, y_pred),
        'precision'           : precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall'              : recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1'                  : f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'auc'                 : roc_auc_score(y_test, y_prob),
        'y_pred'              : y_pred,
        'y_prob'              : y_prob,
        'binarize_thresholds' : thresholds,
    }


# ─────────────────────────────────────────────────────────────
# Cross-validation helper
# ─────────────────────────────────────────────────────────────

def evaluate(X, y, k_values=(5, 10, 15, 20)):
    """
    Run stratified k-fold CV for BNB_nan.
    X may contain NaN — no imputation is applied.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    W = 24
    def _fmt(val, ci_tuple):
        return f"{val:.2f} [{ci_tuple[0]:.2f}, {ci_tuple[1]:.2f}]"

    print(f"\n{'='*125}")
    print(f"  Bernoulli Naive Bayes (NaN-aware)  (skip missing features, no imputation)")
    print(f"{'='*125}")
    print(f"  {'k':<5} {'Accuracy':<{W}} {'Precision':<{W}} {'Recall':<{W}} {'F-Score':<{W}} {'AUC':<{W}}")
    print(f"  {'-'*122}")

    results = {}
    for k in k_values:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        y_true_all, y_pred_all, y_prob_all = [], [], []
        for train_idx, test_idx in cv.split(X, y):
            scores = train_and_score(
                X[train_idx], y[train_idx],
                X[test_idx],  y[test_idx],
            )
            y_true_all.append(y[test_idx])
            y_pred_all.append(scores['y_pred'])
            y_prob_all.append(scores['y_prob'])

        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        y_prob_all = np.concatenate(y_prob_all)

        acc  = accuracy_score(y_true_all, y_pred_all)
        prec = precision_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
        rec  = recall_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
        f1   = f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
        auc  = roc_auc_score(y_true_all, y_prob_all)

        ci = bootstrap_ci(y_true_all, y_pred_all, y_prob_all)

        print(f"  {k:<5} {_fmt(acc,  ci['accuracy']):<{W}} {_fmt(prec, ci['precision']):<{W}}"
              f" {_fmt(rec,  ci['recall']):<{W}} {_fmt(f1,   ci['f1']):<{W}}"
              f" {_fmt(auc,  ci['auc']):<{W}}")

        tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all).ravel()
        results[k] = {
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc,
            'ci': ci,
            'confusion_matrix': {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)},
            'y_true': y_true_all,
            'y_pred': y_pred_all,
            'y_prob': y_prob_all,
        }

    k_cm = 10 if 10 in results else list(results.keys())[-1]
    cm = results[k_cm]['confusion_matrix']
    print(f"\n  Confusion Matrix (k={k_cm}):")
    print(f"    TN={cm['TN']}  FP={cm['FP']}")
    print(f"    FN={cm['FN']}  TP={cm['TP']}")

    return results
