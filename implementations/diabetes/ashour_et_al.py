import os
import sys
import copy
import random
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
# 0. REPRODUCIBILITY
# ─────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
# 2. PREPROCESSING (paper-aligned)
#    - Zero values in clinical columns treated as invalid
#    - Replace them with feature means
#    - Remove duplicates
#    - StandardScaler is applied inside each CV fold
# ─────────────────────────────────────────────
ZERO_AS_MISSING = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

df_clean = df.copy()
for col in ZERO_AS_MISSING:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].replace(0, np.nan)

before_dupes = len(df_clean)
df_clean = df_clean.drop_duplicates().reset_index(drop=True)

print(f"\n=== After paper-style cleaning ===")
print(f"Rows after duplicate removal: {len(df_clean)} (removed {before_dupes - len(df_clean)})")
print(df_clean['Outcome'].value_counts().rename({0: 'Negative', 1: 'Positive'}))

X = df_clean.drop(columns='Outcome').values.astype(np.float32)
y = df_clean['Outcome'].values.astype(np.int64)


# ─────────────────────────────────────────────
# 3. PYTORCH WRAPPERS
# ─────────────────────────────────────────────
class TorchFNN(nn.Module):
    def __init__(self, input_dim, hidden1=8, hidden2=16, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.Tanh(),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Dropout(dropout),

            nn.Linear(hidden2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class TorchCNN(nn.Module):
    def __init__(self, input_len, conv_channels=8, kernel_size=7, dense_units=16, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv1d(1, conv_channels, kernel_size=kernel_size)
        self.act = nn.Tanh()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.drop = nn.Dropout(dropout)

        conv_out_len = input_len - kernel_size + 1
        pool_out_len = conv_out_len // 2
        flattened = conv_channels * pool_out_len

        self.fc1 = nn.Linear(flattened, dense_units)
        self.fc2 = nn.Linear(dense_units, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.drop(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x.squeeze(1)


class BaseTorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        model_type='fnn',
        lr=1e-3,
        batch_size=8,
        dropout=0.2,
        l2=1e-2,
        epochs=100,
        patience=20,
        random_state=42,
        verbose=False,
    ):
        self.model_type = model_type
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout
        self.l2 = l2
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose

    def _build_model(self, input_dim):
        if self.model_type == 'fnn':
            return TorchFNN(
                input_dim=input_dim,
                hidden1=8,
                hidden2=16,
                dropout=self.dropout
            )
        elif self.model_type == 'cnn':
            return TorchCNN(
                input_len=input_dim,
                conv_channels=8,   # not specified explicitly in paper
                kernel_size=7,
                dense_units=16,
                dropout=self.dropout
            )
        raise ValueError(f"Unknown model_type={self.model_type}")

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.25,
            stratify=y,
            random_state=self.random_state
        )

        if self.model_type == 'cnn':
            X_train_t = torch.tensor(X_train).unsqueeze(1)
            X_val_t = torch.tensor(X_val).unsqueeze(1)
        else:
            X_train_t = torch.tensor(X_train)
            X_val_t = torch.tensor(X_val)

        y_train_t = torch.tensor(y_train)
        y_val_t = torch.tensor(y_val)

        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.model_ = self._build_model(X.shape[1]).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.l2
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,
            patience=20,
            min_lr=1e-3
        )

        best_state = None
        best_val_loss = float('inf')
        wait = 0

        for epoch in range(self.epochs):
            self.model_.train()
            for xb, yb in train_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                Xv = X_val_t.to(DEVICE)
                yv = y_val_t.to(DEVICE)
                val_logits = self.model_(Xv)
                val_loss = criterion(val_logits, yv).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model_.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            if self.model_type == 'cnn':
                Xt = torch.tensor(X).unsqueeze(1).to(DEVICE)
            else:
                Xt = torch.tensor(X).to(DEVICE)

            logits = self.model_(Xt)
            probs_pos = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            probs_pos = np.clip(probs_pos, 1e-7, 1 - 1e-7)
            probs_neg = 1.0 - probs_pos
            return np.column_stack([probs_neg, probs_pos])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


# ─────────────────────────────────────────────
# 4. MODELS
# ─────────────────────────────────────────────
models = {
    "Naive Bayes": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", GaussianNB())
    ]),
    "KNN": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier())
    ]),
    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", DecisionTreeClassifier(random_state=SEED))
    ]),
    "J48 (Decision Tree)": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", DecisionTreeClassifier(
            criterion="entropy",
            min_samples_leaf=2,
            random_state=SEED
        ))
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(random_state=SEED))
    ]),
    "FNN": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", BaseTorchClassifier(
            model_type="fnn",
            lr=0.001,
            batch_size=8,
            dropout=0.2,
            l2=0.01,
            epochs=100,
            patience=20,
            random_state=SEED
        ))
    ]),
    "CNN": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", BaseTorchClassifier(
            model_type="cnn",
            lr=0.001,
            batch_size=16,
            dropout=0.2,
            l2=0.001,
            epochs=45,
            patience=20,
            random_state=SEED
        ))
    ]),
}

K_VALUES = [5, 10, 15, 20]


# ─────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────
def evaluate_model(name, model, X, y, k_values):
    W = 24

    def _fmt(val, ci_tuple):
        return f"{val:.2f} [{ci_tuple[0]:.2f}, {ci_tuple[1]:.2f}]"

    print(f"\n{'='*125}")
    print(f" {name}")
    print(f"{'='*125}")
    print(f" {'k':<5} {'Accuracy':<{W}} {'Precision':<{W}} {'Recall':<{W}} {'F-Score':<{W}} {'AUC':<{W}}")
    print(f" {'-'*122}")

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
            f" {k:<5} {_fmt(acc, ci['accuracy']):<{W}} {_fmt(prec, ci['precision']):<{W}}"
            f" {_fmt(rec, ci['recall']):<{W}} {_fmt(f1, ci['f1']):<{W}} {_fmt(auc, ci['auc']):<{W}}"
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
    print(f"\n Confusion Matrix (10-fold):")
    print(f" TN={tn} FP={fp}")
    print(f" FN={fn} TP={tp}")

    return all_results


# Collect results from all models for paired comparison
all_model_results = {}
for name, model in models.items():
    all_model_results[name] = evaluate_model(name, model, X, y, K_VALUES)


# BNB reference
bnb_results = bnb.evaluate(X, y, k_values=K_VALUES)
all_model_results['Bernoulli Naive Bayes'] = bnb_results

# BNB_nan — feed raw data with NaN (no imputation, no scaling)
bnb_nan_results = bnb_nan.evaluate(X, y, k_values=K_VALUES)
all_model_results['BNB (NaN-aware)'] = bnb_nan_results


# ─────────────────────────────────────────────
# 6. DELONG TEST — pairwise AUC comparison vs BNB (k=10)
# ─────────────────────────────────────────────
print(f"\n{'='*80}")
print(" DeLong Test: Pairwise AUC comparison vs Bernoulli Naive Bayes (k=10)")
print(f"{'='*80}")
print(f" {'Model':<25} {'AUC_model':<10} {'AUC_BNB':<10} {'Diff':<10} {'Z':<10} {'p-value':<12} {'Significant?'}")
print(f" {'-'*77}")

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
        f" {name:<25} {result['auc_a']:<10.4f} {result['auc_b']:<10.4f} "
        f"{result['diff']:>+10.4f} {result['z_stat']:>10.4f} {result['p_value']:<12.4f} {sig}"
    )

print(f"\n Note: p < 0.05 indicates statistically significant difference in AUC.")
print(f" 'No' in Significant? column means BNB is not significantly different from that model.")


# ─────────────────────────────────────────────
# 7. McNEMAR TEST — pairwise error-pattern comparison vs BNB (k=10)
# ─────────────────────────────────────────────
print(f"\n{'='*80}")
print(" McNemar Test: Pairwise prediction comparison vs Bernoulli Naive Bayes (k=10)")
print(f"{'='*80}")
print(f" {'Model':<25} {'b (A+B-)':<10} {'c (A-B+)':<10} {'Statistic':<12} {'p-value':<12} {'Method':<8} {'Significant?'}")
print(f" {'-'*87}")

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
        f" {name:<25} {result['b']:<10} {result['c']:<10} {stat_str:<12} "
        f"{result['p_value']:<12.4f} {result['method']:<8} {sig}"
    )

print(f"\n b = instances where the other model is correct but BNB is wrong")
print(f" c = instances where BNB is correct but the other model is wrong")
print(f" H0: both classifiers make errors on the same instances (b = c)")
print(f" p < 0.05 -> significantly different error patterns")


# ─────────────────────────────────────────────
# 8. PAIRED BOOTSTRAP DIFFERENCE CIs - BNB minus comparator (k=10)
# ─────────────────────────────────────────────
print(f"\n{'='*104}")
print(" Paired Bootstrap 95% CIs: Bernoulli Naive Bayes minus comparator (k=10)")
print(f"{'='*104}")
print(f" {'Model':<25} {'Acc diff [95% CI]':<24} {'F1 diff [95% CI]':<24} {'AUC diff [95% CI]':<24}")
print(f" {'-'*101}")

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
        f" {name:<25} "
        f"{_fmt_diff('accuracy'):<24} "
        f"{_fmt_diff('f1'):<24} "
        f"{_fmt_diff('auc'):<24}"
    )

print(f"\n Positive differences favor Bernoulli Naive Bayes.")
print(f" If a 95% CI includes 0, the paired bootstrap does not show a clear difference on that metric.")


# ─────────────────────────────────────────────
# 9. DIRECT: BNB (NaN-aware) vs BNB (mean imputed) — k=10
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
print(f" Direct comparison: BNB (NaN-aware) vs BNB (mean imputed) — k={k_ref}")
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
# 10. DELONG TEST — pairwise AUC comparison vs BNB_nan (k=10)
# ─────────────────────────────────────────────
print(f"\n{'='*80}")
print(" DeLong Test: Pairwise AUC comparison vs BNB (NaN-aware) (k=10)")
print(f"{'='*80}")
print(f" {'Model':<25} {'AUC_model':<10} {'AUC_BNBnan':<10} {'Diff':<10} {'Z':<10} {'p-value':<12} {'Significant?'}")
print(f" {'-'*77}")

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
        f" {name:<25} {result['auc_a']:<10.4f} {result['auc_b']:<10.4f} "
        f"{result['diff']:>+10.4f} {result['z_stat']:>10.4f} {result['p_value']:<12.4f} {sig}"
    )

print(f"\n Note: p < 0.05 indicates statistically significant difference in AUC.")
print(f" 'No' means BNB (NaN-aware) is not significantly different from that model.")


# ─────────────────────────────────────────────
# 11. McNEMAR TEST — pairwise error-pattern comparison vs BNB_nan (k=10)
# ─────────────────────────────────────────────
print(f"\n{'='*80}")
print(" McNemar Test: Pairwise prediction comparison vs BNB (NaN-aware) (k=10)")
print(f"{'='*80}")
print(f" {'Model':<25} {'b (A+B-)':<10} {'c (A-B+)':<10} {'Statistic':<12} {'p-value':<12} {'Method':<8} {'Significant?'}")
print(f" {'-'*87}")

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
        f" {name:<25} {result['b']:<10} {result['c']:<10} {stat_str:<12} "
        f"{result['p_value']:<12.4f} {result['method']:<8} {sig}"
    )

print(f"\n b = instances where the other model is correct but BNB (NaN-aware) is wrong")
print(f" c = instances where BNB (NaN-aware) is correct but the other model is wrong")
print(f" H0: both classifiers make errors on the same instances (b = c)")
print(f" p < 0.05 -> significantly different error patterns")


# ─────────────────────────────────────────────
# 12. PAIRED BOOTSTRAP DIFFERENCE CIs — BNB_nan minus comparator (k=10)
# ─────────────────────────────────────────────
print(f"\n{'='*104}")
print(" Paired Bootstrap 95% CIs: BNB (NaN-aware) minus comparator (k=10)")
print(f"{'='*104}")
print(f" {'Model':<25} {'Acc diff [95% CI]':<24} {'F1 diff [95% CI]':<24} {'AUC diff [95% CI]':<24}")
print(f" {'-'*101}")

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
        f" {name:<25} "
        f"{_fmt_nan_diff('accuracy'):<24} "
        f"{_fmt_nan_diff('f1'):<24} "
        f"{_fmt_nan_diff('auc'):<24}"
    )

print(f"\n Positive differences favor BNB (NaN-aware).")
print(f" If a 95% CI includes 0, the paired bootstrap does not show a clear difference on that metric.")

print("\n=== Done ===")
