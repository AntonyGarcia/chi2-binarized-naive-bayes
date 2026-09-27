"""
Run a comparison script unchanged and record the full-precision p-value of every
DeLong and McNemar test it performs (used for the multiple-comparison correction).

Usage: python method_evaluation/capture_test_pvalues.py <script.py> <output.json>
The script's normal output is printed as usual.
"""

import json
import os
import runpy
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALGO_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "algorithms"))
sys.path.insert(0, ALGO_DIR)

import bnb  # noqa: E402

CALLS = []


def _record(kind, fn):
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        CALLS.append({"test": kind, "p_value": float(result["p_value"])})
        return result
    return wrapper


bnb.delong_test = _record("DeLong", bnb.delong_test)
bnb.mcnemar_test = _record("McNemar", bnb.mcnemar_test)

# Also record the AUC and its bootstrap CI at full precision (some scripts print 2 decimals).
_bootstrap_ci = bnb.bootstrap_ci


def _record_ci(y_true, y_pred, y_prob, *args, **kwargs):
    from sklearn.metrics import roc_auc_score

    ci = _bootstrap_ci(y_true, y_pred, y_prob, *args, **kwargs)
    CALLS.append({"test": "AUC", "auc": float(roc_auc_score(y_true, y_prob)),
                  "ci": [float(ci["auc"][0]), float(ci["auc"][1])]})
    return ci


bnb.bootstrap_ci = _record_ci

if __name__ == "__main__":
    script, out_json = sys.argv[1], sys.argv[2]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    sys.argv = [script]
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(CALLS, fh, indent=1)
