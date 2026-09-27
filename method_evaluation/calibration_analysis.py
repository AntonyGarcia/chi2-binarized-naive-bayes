"""
Calibration of chi2-binarized Naive Bayes (Table 8 and Figure 1 of the paper).

Protocol (leakage-safe, stratified 10-fold CV, seed 42):
  1. In each fold, chi2 thresholds and the Naive Bayes model are fitted on the
     training split only, exactly as in the main evaluation pipeline
     (bnb_binarization_variants_three_datasets.fit_binarized_model).
  2. A three-parameter beta calibration map is fitted on the training-split
     predictions and applied to the held-out split.
  3. Raw and calibrated out-of-fold probabilities are pooled and summarised by
     Brier score, calibration intercept and slope (logistic recalibration),
     observed event rate and mean predicted risk.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
ALGO_DIR = os.path.join(ROOT_DIR, "algorithms")
for path in (BASE_DIR, ALGO_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import bnb_binarization_variants_three_datasets as variants
from bnb import encode_categorical


SEED = 42
K = 10
EPS = 1e-15
N_BINS = 10
OUT_DIR = os.path.join(BASE_DIR, "outputs")
CHI2 = variants.SPLIT_METHODS["chi2"]


def _logit(p, eps=EPS):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


class BetaCalibration:
    """Three-parameter beta calibration (Kull et al., 2017) with a, b >= 0."""

    def fit(self, p, y):
        p = np.clip(p, EPS, 1 - EPS)
        Z = np.column_stack([np.log(p), -np.log(1 - p)])
        keep = [0, 1]
        while True:
            lr = LogisticRegression(penalty=None, max_iter=10000).fit(Z[:, keep], y)
            coef = dict(zip(keep, lr.coef_[0]))
            negative = [j for j, c in coef.items() if c < 0]
            if not negative or len(keep) == 1:
                break
            keep = [j for j in keep if j not in negative]
        self.keep_, self.model_ = keep, lr
        return self

    def predict(self, p):
        p = np.clip(p, EPS, 1 - EPS)
        Z = np.column_stack([np.log(p), -np.log(1 - p)])
        return self.model_.predict_proba(Z[:, self.keep_])[:, 1]


def calibration_metrics(y, p):
    """Brier score, calibration intercept and slope, event rate and mean risk."""
    lp = _logit(p).reshape(-1, 1)
    slope_model = LogisticRegression(penalty=None, max_iter=10000).fit(lp, y)
    return {
        "Brier": float(np.mean((p - y) ** 2)),
        "Intercept": float(slope_model.intercept_[0]),
        "Slope": float(slope_model.coef_[0][0]),
        "Event rate": float(np.mean(y)),
        "Mean risk": float(np.mean(p)),
    }


def cross_validated_probabilities(ds):
    X, y = ds["X"], ds["y"]
    categorical_idx = ds.get("categorical_idx")
    n_categories = None
    if categorical_idx:
        X, n_categories = encode_categorical(X, categorical_idx)

    raw = np.zeros(len(y))
    cal = np.zeros(len(y))
    cv = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    for train_idx, test_idx in cv.split(X, y):
        model, X_test_bin = variants.fit_binarized_model(
            X[train_idx], y[train_idx], X[test_idx], CHI2,
            ds["use_nan_aware"], categorical_idx, n_categories,
        )
        # Training-split predictions for the calibration map.
        _, X_train_bin = variants.fit_binarized_model(
            X[train_idx], y[train_idx], X[train_idx], CHI2,
            ds["use_nan_aware"], categorical_idx, n_categories,
        )
        pos = list(model.classes_).index(1)
        p_train = model.predict_proba(X_train_bin)[:, pos]
        p_test = model.predict_proba(X_test_bin)[:, pos]

        raw[test_idx] = p_test
        cal[test_idx] = BetaCalibration().fit(p_train, y[train_idx]).predict(p_test)
    return y, raw, cal


def reliability_curve(y, p, n_bins=N_BINS):
    """Observed proportion vs mean predicted probability in quantile bins."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    xs, ys = [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            xs.append(p[m].mean())
            ys.append(y[m].mean())
    return np.array(xs), np.array(ys)


def plot_curves(curves, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(curves), 1, figsize=(6.5, 5.8 * len(curves)))
    for ax, (letter, (y, raw, cal)) in zip(axes, curves.items()):
        ax.plot([0, 1], [0, 1], color="0.8", label="Perfect (intercept=0; slope=1)")
        xr, yr = reliability_curve(y, raw)
        xc, yc = reliability_curve(y, cal)
        ax.plot(xr, yr, "k--s", markersize=5, label="Raw probabilities")
        ax.plot(xc, yc, "k-o", markersize=5, label="Beta-calibrated probabilities")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed proportion")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="upper left")
        ax.text(-0.2, 1.0, f"{letter})", transform=ax.transAxes, fontsize=30, family="serif", va="top")
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--figure", default=os.path.join(OUT_DIR, "calibration_curves.png"))
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    # Panel order in the paper: (a) Pima, (b) Heart Failure, (c) Wisconsin.
    datasets = {
        "a": ("Pima Diabetes", variants.load_diabetes()),
        "b": ("Heart Disease", variants.load_heart_disease()),
        "c": ("Breast Cancer", variants.load_breast_cancer()),
    }
    rows, curves = [], {}
    for letter, (label, ds) in datasets.items():
        y, raw, cal = cross_validated_probabilities(ds)
        curves[letter] = (y, raw, cal)
        for model_name, p in (("Raw", raw), ("Beta-calibrated", cal)):
            rows.append({"Dataset": label, "Model": model_name, **calibration_metrics(y, p)})

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT_DIR, "calibration_table.csv"), index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    plot_curves(curves, args.figure)
    print(f"\nFigure saved: {args.figure}")


if __name__ == "__main__":
    main()
