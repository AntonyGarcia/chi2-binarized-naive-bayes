"""Full-precision DeLong and McNemar p-values of Table 4 (each variant vs chi2), saved as JSON."""

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "..", "algorithms"))

import bnb_binarization_variants_three_datasets as variants  # noqa: E402
from bnb import delong_test, mcnemar_test  # noqa: E402

out = {}
for ds in (variants.load_diabetes(), variants.load_breast_cancer(), variants.load_heart_disease()):
    results = {
        name: variants.evaluate_variant(ds["X"], ds["y"], method, ds["use_nan_aware"], ds.get("categorical_idx"))
        for name, method in variants.SPLIT_METHODS.items()
    }
    base = results["chi2"]
    out[ds["name"]] = {
        name: {
            "DeLong": float(delong_test(base["y_true"], r["y_prob"], base["y_prob"])["p_value"]),
            "McNemar": float(mcnemar_test(base["y_true"], r["y_pred"], base["y_pred"])["p_value"]),
        }
        for name, r in results.items() if name != "chi2"
    }
path = os.path.join(BASE_DIR, "outputs", "pvalues", "table4.json")
with open(path, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1))
