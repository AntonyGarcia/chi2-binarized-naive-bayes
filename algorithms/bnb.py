"""
Bernoulli Naive Bayes — standalone control algorithm.

Binarization strategy
---------------------
For each feature, the optimal binarization threshold is selected from the
training fold only (no leakage):
  1. Collect all unique values of the feature in the training fold.
  2. For each candidate threshold t, binarize the feature as (x > t).
  3. Build a 2×2 contingency table against the binary outcome y.
  4. Apply a chi-square test; retain the threshold with the lowest p-value.
The selected per-feature thresholds are then used to binarize both the
training and test folds before fitting/predicting.

Interface
---------
chi2_threshold(x_col, y) -> float
    Select the best binarization threshold for a single feature column.

train_and_score(X_train, y_train, X_test, y_test) -> dict
    Fit on training fold, score on test fold.
    Returns: accuracy, precision, recall, f1, auc, y_pred, y_prob, binarize_thresholds

evaluate(X, y, k_values, categorical_idx=None)
    Run stratified k-fold CV for each k and print a results table.
    Metrics are computed from all concatenated fold predictions (overall, not mean-of-folds).
    AUC is computed once from all fold probability scores.

Categorical features
--------------------
When categorical_idx is given, only the remaining (continuous) columns are
binarized; categorical columns keep their original categories and the model
is a Categorical Naive Bayes over [binarized continuous | categorical] columns.
With Laplace smoothing (alpha = 1) the binarized columns get exactly the
Bernoulli NB estimates (n + 1) / (N + 2), so this reduces to Bernoulli NB when
there are no categorical columns.
"""

import numpy as np
from scipy.stats import chi2_contingency
from sklearn.naive_bayes import BernoulliNB, CategoricalNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)


# ─────────────────────────────────────────────────────────────
# Bootstrap confidence intervals
# ─────────────────────────────────────────────────────────────

def bootstrap_ci(y_true, y_pred, y_prob, n_boot=1000, ci=95, random_state=42):
    """
    Compute bootstrap confidence intervals for all five metrics.

    Resamples (y_true, y_pred, y_prob) with replacement n_boot times and
    returns the (lo, hi) percentile bounds at the requested CI level.

    Parameters
    ----------
    y_true, y_pred, y_prob : 1-D ndarrays  Ground truth, predictions, probabilities.
    n_boot      : int   Number of bootstrap iterations (default 1000).
    ci          : float Confidence level in percent (default 95).
    random_state: int   Seed for reproducibility.

    Returns
    -------
    dict mapping metric name -> (lower, upper) floats.
    """
    rng  = np.random.default_rng(random_state)
    n    = len(y_true)
    lo_p = (100 - ci) / 2
    hi_p = 100 - lo_p

    accs, precs, recs, f1s, aucs = [], [], [], [], []
    for _ in range(n_boot):
        idx   = rng.integers(0, n, size=n)
        yt    = y_true[idx]
        yp    = y_pred[idx]
        ypr   = y_prob[idx]
        accs.append(accuracy_score(yt, yp))
        precs.append(precision_score(yt, yp, average='weighted', zero_division=0))
        recs.append(recall_score(yt, yp, average='weighted', zero_division=0))
        f1s.append(f1_score(yt, yp, average='weighted', zero_division=0))
        if len(np.unique(yt)) == 2:
            aucs.append(roc_auc_score(yt, ypr))

    def _ci(arr):
        return (float(np.percentile(arr, lo_p)), float(np.percentile(arr, hi_p)))

    return {
        'accuracy' : _ci(accs),
        'precision': _ci(precs),
        'recall'   : _ci(recs),
        'f1'       : _ci(f1s),
        'auc'      : _ci(aucs) if aucs else (float('nan'), float('nan')),
    }


def paired_bootstrap_diff_ci(
    y_true,
    y_pred_a,
    y_prob_a,
    y_pred_b,
    y_prob_b,
    n_boot=2000,
    ci=95,
    random_state=42,
):
    """
    Compute paired stratified bootstrap CIs for metric differences (A - B).

    Both models are resampled on the exact same bootstrap patients, preserving
    the pairing required for fair model comparison. Positives and negatives are
    resampled separately to keep class balance stable across bootstrap draws.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_prob_a = np.asarray(y_prob_a)
    y_pred_b = np.asarray(y_pred_b)
    y_prob_b = np.asarray(y_prob_b)

    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    lo_p = (100 - ci) / 2
    hi_p = 100 - lo_p

    diffs = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': [],
    }

    for _ in range(n_boot):
        sample_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        sample_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        sample_idx = np.concatenate([sample_pos, sample_neg])
        rng.shuffle(sample_idx)

        yt = y_true[sample_idx]
        yp_a = y_pred_a[sample_idx]
        ypr_a = y_prob_a[sample_idx]
        yp_b = y_pred_b[sample_idx]
        ypr_b = y_prob_b[sample_idx]

        diffs['accuracy'].append(accuracy_score(yt, yp_a) - accuracy_score(yt, yp_b))
        diffs['precision'].append(
            precision_score(yt, yp_a, average='weighted', zero_division=0)
            - precision_score(yt, yp_b, average='weighted', zero_division=0)
        )
        diffs['recall'].append(
            recall_score(yt, yp_a, average='weighted', zero_division=0)
            - recall_score(yt, yp_b, average='weighted', zero_division=0)
        )
        diffs['f1'].append(
            f1_score(yt, yp_a, average='weighted', zero_division=0)
            - f1_score(yt, yp_b, average='weighted', zero_division=0)
        )
        diffs['auc'].append(roc_auc_score(yt, ypr_a) - roc_auc_score(yt, ypr_b))

    observed = {
        'accuracy': accuracy_score(y_true, y_pred_a) - accuracy_score(y_true, y_pred_b),
        'precision': (
            precision_score(y_true, y_pred_a, average='weighted', zero_division=0)
            - precision_score(y_true, y_pred_b, average='weighted', zero_division=0)
        ),
        'recall': (
            recall_score(y_true, y_pred_a, average='weighted', zero_division=0)
            - recall_score(y_true, y_pred_b, average='weighted', zero_division=0)
        ),
        'f1': (
            f1_score(y_true, y_pred_a, average='weighted', zero_division=0)
            - f1_score(y_true, y_pred_b, average='weighted', zero_division=0)
        ),
        'auc': roc_auc_score(y_true, y_prob_a) - roc_auc_score(y_true, y_prob_b),
    }

    intervals = {
        metric: (
            float(np.percentile(values, lo_p)),
            float(np.percentile(values, hi_p)),
        )
        for metric, values in diffs.items()
    }

    return {
        'observed_diff': observed,
        'ci': intervals,
    }


# ─────────────────────────────────────────────────────────────
# DeLong test for comparing two correlated AUCs
# ─────────────────────────────────────────────────────────────

def _compute_midrank(x):
    """Compute midranks for an array (handles ties)."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def _structural_components(y_true, y_prob):
    """
    Compute structural components (placement values) for DeLong test.
    V10[i] = fraction of negatives with score < pos_i (+ 0.5 * ties)
    V01[j] = fraction of positives with score > neg_j (+ 0.5 * ties)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    scores_pos = y_prob[pos_mask]
    scores_neg = y_prob[neg_mask]
    m = len(scores_pos)
    n = len(scores_neg)
    V10 = np.zeros(m)
    for i, sp in enumerate(scores_pos):
        V10[i] = (np.sum(scores_neg < sp) + 0.5 * np.sum(scores_neg == sp)) / n
    V01 = np.zeros(n)
    for j, sn in enumerate(scores_neg):
        V01[j] = (np.sum(scores_pos > sn) + 0.5 * np.sum(scores_pos == sn)) / m
    return V10, V01


def delong_test(y_true, y_prob_a, y_prob_b):
    """DeLong test comparing two correlated AUCs."""
    from scipy.stats import norm
    y_true = np.asarray(y_true)
    y_prob_a = np.asarray(y_prob_a)
    y_prob_b = np.asarray(y_prob_b)
    m = (y_true == 1).sum()
    n = (y_true == 0).sum()
    V10_a, V01_a = _structural_components(y_true, y_prob_a)
    V10_b, V01_b = _structural_components(y_true, y_prob_b)
    auc_a = V10_a.mean()
    auc_b = V10_b.mean()
    S10 = np.cov(np.vstack([V10_a, V10_b]), ddof=1)
    S01 = np.cov(np.vstack([V01_a, V01_b]), ddof=1)
    S = S10 / m + S01 / n
    diff = auc_a - auc_b
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    se = np.sqrt(max(var_diff, 1e-12))
    z = diff / se
    p = 2 * norm.sf(abs(z))

    return {'auc_a': auc_a, 'auc_b': auc_b, 'diff': diff, 'z_stat': z, 'p_value': p}


# ─────────────────────────────────────────────────────────────
# McNemar's test for comparing two classifiers' error patterns
# ─────────────────────────────────────────────────────────────

def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """
    McNemar's test comparing two classifiers on the same instances.

    Builds a 2×2 contingency table of discordant predictions:
        b = A correct, B wrong
        c = A wrong,   B correct

    Uses the chi-square form with continuity correction when b+c >= 25,
    otherwise falls back to an exact binomial test.

    Parameters
    ----------
    y_true   : 1-D array  Ground truth labels.
    y_pred_a : 1-D array  Predictions from model A.
    y_pred_b : 1-D array  Predictions from model B.

    Returns
    -------
    dict with keys:
        b, c       : int    Discordant counts.
        statistic  : float  Chi-square statistic (or NaN for exact test).
        p_value    : float  Two-sided p-value.
        method     : str    'chi2' or 'exact'.
    """
    from scipy.stats import binom

    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    b = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))   # A wrong, B right

    n = b + c
    if n == 0:
        return {'b': b, 'c': c, 'statistic': 0.0, 'p_value': 1.0, 'method': 'chi2'}

    if n >= 25:
        chi2 = (abs(b - c) - 1) ** 2 / n
        from scipy.stats import chi2 as chi2_dist
        p = chi2_dist.sf(chi2, df=1)
        return {'b': b, 'c': c, 'statistic': chi2, 'p_value': p, 'method': 'chi2'}
    else:
        # Exact binomial test: P(X >= max(b,c)) under H0: p=0.5
        k = max(b, c)
        p = 2 * binom.sf(k - 1, n, 0.5)
        p = min(p, 1.0)
        return {'b': b, 'c': c, 'statistic': float('nan'), 'p_value': p, 'method': 'exact'}


# ─────────────────────────────────────────────────────────────
# Threshold selection via chi-square test
# ─────────────────────────────────────────────────────────────

def chi2_threshold(x_col, y):
    """
    Select the best binarization threshold for a single feature column.

    For every unique value in x_col (sorted), binarize the column as (x > t)
    and compute a chi-square test against the binary outcome y.  The threshold
    with the lowest p-value is returned.  In case of ties the first (lowest)
    candidate is chosen.

    Parameters
    ----------
    x_col : 1-D array  Continuous feature values (training fold only).
    y     : 1-D array  Binary outcome labels.

    Returns
    -------
    float  Best binarization threshold.
    """
    candidates = np.unique(x_col)
    best_t, best_p = candidates[0], 1.0
    for t in candidates:
        binary = (x_col > t).astype(int)
        # 2×2 contingency table: rows = binarized feature, cols = outcome
        table = np.zeros((2, 2), dtype=int)
        for b, label in zip(binary, y):
            table[int(b), int(label)] += 1
        # Skip degenerate splits (all-zero row/col would break chi2)
        if table.min() == 0 and (table[0].sum() == 0 or table[1].sum() == 0):
            continue
        _, p, _, _ = chi2_contingency(table)
        if p < best_p:
            best_p, best_t = p, t
    return best_t


# ─────────────────────────────────────────────────────────────
# Core function — one train/test split
# ─────────────────────────────────────────────────────────────

HEART_CATEGORICAL = ("Sex", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope")


def categorical_indices(feature_names, categorical_names=HEART_CATEGORICAL):
    """Column indices of the categorical features present in feature_names."""
    return [j for j, name in enumerate(feature_names) if name in categorical_names]


def encode_categorical(X, categorical_idx):
    """
    Replace each categorical column by integer codes 0..r-1.

    Uses only the observed category values (no outcome information), so it can
    be applied to the full dataset before cross-validation. Any monotone
    rescaling of label-encoded columns maps to the same codes.

    Returns the encoded copy of X and the number of categories per column
    (2 for columns that will be binarized).
    """
    X = np.array(X, dtype=float, copy=True)
    n_categories = np.full(X.shape[1], 2, dtype=int)
    for j in categorical_idx:
        uniq, codes = np.unique(X[:, j], return_inverse=True)
        X[:, j] = codes
        n_categories[j] = len(uniq)
    return X, n_categories


def _train_and_score_categorical(X_train, y_train, X_test, categorical_idx, n_categories):
    """Binarize continuous columns (chi2, training fold only) and fit Categorical NB."""
    cat = np.zeros(X_train.shape[1], dtype=bool)
    cat[list(categorical_idx)] = True

    thresholds = np.full(X_train.shape[1], np.nan)
    for j in np.where(~cat)[0]:
        thresholds[j] = chi2_threshold(X_train[:, j], y_train)

    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()
    X_train_enc[:, ~cat] = X_train[:, ~cat] > thresholds[~cat]
    X_test_enc[:, ~cat] = X_test[:, ~cat] > thresholds[~cat]

    model = CategoricalNB(alpha=1.0, min_categories=n_categories)
    model.fit(X_train_enc.astype(int), y_train)
    return model, X_test_enc.astype(int), thresholds


def train_and_score(X_train, y_train, X_test, y_test, categorical_idx=None, n_categories=None):
    """
    Fit BernoulliNB on (X_train, y_train) and evaluate on (X_test, y_test).

    Per-feature binarization thresholds are chosen from the training fold via
    chi-square test (lowest p-value across all unique candidate thresholds).
    The same thresholds are applied to binarize the test fold.

    Parameters
    ----------
    X_train, y_train : array-like  Training data and labels.
    X_test,  y_test  : array-like  Test data and labels.

    Returns
    -------
    dict with keys:
        accuracy, precision, recall, f1, auc  — scalar floats
        y_pred               — predicted class labels (ndarray)
        y_prob               — predicted positive-class probabilities (ndarray)
        binarize_thresholds  — per-feature chi2-selected thresholds (ndarray)
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test  = np.asarray(X_test,  dtype=float)

    if categorical_idx:
        model, X_test_bin, thresholds = _train_and_score_categorical(
            X_train, y_train, X_test, categorical_idx, n_categories)
    else:
        # Select per-feature threshold from training fold only (no leakage)
        thresholds = np.array([chi2_threshold(X_train[:, j], y_train)
                               for j in range(X_train.shape[1])])

        X_train_bin = (X_train > thresholds).astype(float)
        X_test_bin  = (X_test  > thresholds).astype(float)

        model = BernoulliNB(binarize=None)   # data already binarized
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

def evaluate(X, y, k_values=(5, 10, 15, 20), categorical_idx=None):
    """
    Run stratified k-fold CV for each value in k_values and print a results table.

    Binarization threshold is computed per fold inside train_and_score (per-feature
    chi-square-selected threshold from that fold's training split), so no test-fold
    information leaks.
    Metrics are computed from all concatenated fold predictions (overall metrics,
    not mean-of-fold-metrics).  AUC is computed once from the full set of
    held-out probability scores.

    Parameters
    ----------
    X, y       : array-like  Feature matrix and label vector.
    k_values   : iterable    Fold counts to evaluate (default: 5,10,15,20).
    categorical_idx : list of int or None
                 Columns kept as categories (Categorical NB); the rest are binarized.

    Returns
    -------
    results : dict[int, dict]  Maps each k to its metrics dict (+ confusion matrix).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n_categories = None
    if categorical_idx:
        X, n_categories = encode_categorical(X, categorical_idx)

    W = 24   # column width per metric
    def _fmt(val, ci_tuple):
        return f"{val:.2f} [{ci_tuple[0]:.2f}, {ci_tuple[1]:.2f}]"

    print(f"\n{'='*125}")
    print(f"  Bernoulli Naive Bayes  (binarize=chi2 threshold per feature, per training fold)")
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
                categorical_idx=categorical_idx, n_categories=n_categories,
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
            # Store predictions for paired comparison (DeLong test)
            'y_true': y_true_all,
            'y_pred': y_pred_all,
            'y_prob': y_prob_all,
        }

    # Confusion matrix for k=10
    k_cm = 10 if 10 in results else list(results.keys())[-1]
    cm = results[k_cm]['confusion_matrix']
    print(f"\n  Confusion Matrix (k={k_cm}):")
    print(f"    TN={cm['TN']}  FP={cm['FP']}")
    print(f"    FN={cm['FN']}  TP={cm['TP']}")

    return results


# ─────────────────────────────────────────────────────────────
# Standalone entry-point — run on the diabetes dataset
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    import pandas as pd

    _BASE = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(_BASE, '..', 'datasets', 'diabetes.csv'))

    ZERO_AS_MISSING = ['Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI']
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, float('nan'))
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"Instances after preprocessing: {len(df)}")
    print(df['Outcome'].value_counts().rename({0: 'Negative', 1: 'Positive'}))

    X = df.drop(columns='Outcome').values
    y = df['Outcome'].values

    evaluate(X, y, k_values=[5, 10, 15, 20])

    print("\n=== Done ===")
