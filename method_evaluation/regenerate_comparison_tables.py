"""
Regenerate the AUC and p-value cells of Tables 4-7 of the manuscript from the outputs of
the comparison scripts (captured with capture_test_pvalues.py), applying the
Benjamini-Hochberg correction separately for each dataset and each test.

Usage: python method_evaluation/regenerate_comparison_tables.py <manuscript.tex>
"""

import json
import os
import re
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from parse_comparison_output import parse  # noqa: E402

PV = os.path.join(BASE_DIR, "outputs", "pvalues")

# Table blocks: (cite key, script, reference in the test-section title,
#                {table row: script model name}, table row of the reference model)
BLOCKS = {
    "tab:diabetes_statistical_comparison": [
        ("ashourOptimizedNeuralNetworks2024", "ashour_et_al", "BNB (NaN-aware)", {
            "Gaussian Naïve Bayes": "Naive Bayes", "k-Nearest Neighbors": "KNN",
            "J48 Decision Tree": "J48 (Decision Tree)", "Random Forest": "Random Forest",
            "Feedforward Neural Network": "FNN", "Convolutional Neural Network": "CNN",
            "Bernoulli Naïve Bayes": "Bernoulli Naive Bayes"}, "Bernoulli Naïve Bayes (NaN-aware)"),
        ("battineniComparativeMachineLearningApproach2019", "battineni_et_al", "Bernoulli Naive Bayes", {
            "Gaussian Naïve Bayes": "Naive Bayes", "J48 Decision Tree": "J48 (Decision Tree)",
            "Random Forest": "Random Forest", "Logistic Regression": "Logistic Regression"},
         "Bernoulli Naïve Bayes"),
        ("bhoi2021prediction", "bhoi_et_al", "BNB", {
            "Classification Tree": "CT (Classification Tree)", "Support Vector Machine": "SVM",
            "k-Nearest Neighbors": "k-NN", "Gaussian Naïve Bayes": "NB (GaussianNB)",
            "Random Forest": "RF (Random Forest)", "Multilayer Perceptron": "NN (MLP)",
            "AdaBoost": "AB (AdaBoost)", "Logistic Regression": "LR (Logistic Regression)"},
         "Bernoulli Naïve Bayes"),
        ("prantoEvaluatingMachineLearning2020", "pranto_et_al", "BNB (NaN-aware)", {
            "Gaussian Naïve Bayes": "Gaussian Naive Bayes", "Decision Tree": "Decision Tree",
            "Random Forest": "Random Forest", "k-Nearest Neighbors": "KNN",
            "Bernoulli Naïve Bayes": "Bernoulli Naive Bayes"}, "Bernoulli Naïve Bayes (NaN-aware)"),
        ("REZA2023100118", "reza_et_al", "BNB (NaN-aware)", {
            "SVM (Linear)": "SVM - Linear", "SVM (Polynomial)": "SVM - Polynomial",
            "SVM (RBF Minkowski)": "SVM - RBF Minkowski", "SVM (RBF City Block)": "SVM - RBF City Block",
            "SVM (RBF Mahalanobis)": "SVM - RBF Mahalanobis", "SVM (RBF Bray-Curtis)": "SVM - RBF Bray-Curtis",
            "SVM (RBF Canberra)": "SVM - RBF Canberra", "SVM (RBF Euclidean Distance)": "SVM - RBF standardized ED",
            "Bernoulli Naïve Bayes": "Bernoulli Naive Bayes"}, "Bernoulli Naïve Bayes (NaN-aware)"),
    ],
    "tab:cancer_statistical_comparison": [
        ("azarProbabilisticNeuralNetwork2013", "azar_et_al", "Bernoulli Naive Bayes", "SAME", "Bernoulli Naïve Bayes"),
        ("jakharSELFStackedbasedEnsemble2023", "jakhar_et_al", "Bernoulli Naive Bayes", "SAME", "Bernoulli Naïve Bayes"),
        ("NAJI2021487", "naji_et_al", "Bernoulli Naive Bayes", "SAME", "Bernoulli Naïve Bayes"),
        ("rovshenovPerformanceComparisonDifferent2022", "roshenov", "Bernoulli Naive Bayes", "SAME", "Bernoulli Naïve Bayes"),
        ("shahMachineLearningTechniques2022", "shah_et_al", "Bernoulli Naive Bayes", "SAME", "Bernoulli Naïve Bayes"),
    ],
    "tab:heart_statistical_comparison": [
        ("kumarmygapulaPerformanceEvaluationMachine2023", "kumar_et_al", "Bernoulli Naive Bayes", {
            "SVM": "Support Vector Machine (SVM)", "Random Forest": "Random Forest", "Naïve Bayes": "Naive Bayes",
            "Logistic Regression": "Logistic Regression", "KNN": "K-Nearest Neighbors (KNN)",
            "Decision Tree": "Decision Tree"}, "Bernoulli Naïve Bayes"),
        ("liuPredictiveClassifierCardiovascular2022", "liu_et_al", "Bernoulli Naive Bayes", {
            "Logistic Regression (LR)": "Logistic Regression", "Random Forest (RF)": "Random Forest",
            "Extra Trees (ET)": "Extra Trees", "GBDT": "CatBoost-style GBDT Proxy",
            "Multilayer Perceptron (MLP)": "Multilayer Perceptron (MLP)",
            "Stacking Ensemble": "Stacking Ensemble (Liu et al.)"}, "Bernoulli Naïve Bayes"),
        ("muhammadEnhancingPrognosisAccuracy2023", "muhammad_et_al", "Bernoulli Naive Bayes", {
            "Logistic Regression": "Logistic Regression", "Decision Tree": "Decision Tree",
            "Random Forest": "Random Forest", "Support Vector Machine (RBF)": "Support Vector Machine (SVM)",
            "K Nearest Neighbors": "K-Nearest Neighbors (KNN)", "Gaussian Naïve Bayes": "Gaussian Naive Bayes"},
         "Bernoulli Naïve Bayes"),
        ("patidarComparativeAnalysisMachine2022", "patidar", "Bernoulli Naive Bayes", {
            "Random Forest": "Random Forest", "Logistic Regression": "Logistic Regression",
            "K-Nearest Neighbours (K=13)": "K-Nearest Neighbors (KNN)"}, "Bernoulli Naïve Bayes"),
    ],
}


def bh(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty(m)
    run = 1.0
    for k, i in enumerate(order):
        run = min(run, p[i] * m / (m - k))
        adj[i] = run
    return adj


def strip(cell):
    """Cell text without highlight wrappers, for comparison."""
    c = cell.strip()
    for macro in ("\\revboxC{", "\\revboxB{", "\\revbox{"):
        while macro in c:
            i = c.index(macro)
            depth, j = 1, i + len(macro)
            while depth:
                depth += {"{": 1, "}": -1}.get(c[j], 0)
                j += 1
            c = c[:i] + c[i + len(macro):j - 1] + c[j:]
    return c.strip()


def mark(old, new):
    return old.strip() if strip(old) == new else f"\\revbox{{{new}}}"


def fmt_auc(auc):
    return f"{auc[0]:.3f} [{auc[1]:.3f},{auc[2]:.3f}]"


def fmt_p(p, better, decimals):
    s = f"<{10 ** -decimals:.{decimals}f}" if p < 10 ** -decimals else f"{p:.{decimals}f}"
    if p < 0.05:
        return f"\\textbf{{{s}$^{{{'*' if better else '**'}}}$}}"
    return s


def block_data(script, reference):
    """k=10 AUC [CI] per model, and the rows of the test sections against `reference`."""
    txt, js = os.path.join(PV, script + ".txt"), os.path.join(PV, script + ".json")
    aucs, sections = parse(txt, js)
    calls = json.load(open(js, encoding="utf-8"))
    auc_calls = [c for c in calls if c["test"] == "AUC"]
    tests = {}
    ref_auc = None
    for sec in sections:
        if sec["reference"] != reference:
            continue
        for row in sec["rows"]:
            tests.setdefault(row["name"], {})[sec["test"]] = (row["p"], row["better"])
    # Exact AUCs: match the AUC printed in the DeLong rows to the recorded bootstrap calls.
    text = open(txt, encoding="utf-8", errors="replace").read()
    m = re.search(r"DeLong Test: .*?vs " + re.escape(reference) + r"\s*\(k=10\).*?\n\s*-{10,}\n(.*?)\n\s*\n",
                  text, re.S)
    exact = {}
    for line in m.group(1).splitlines():
        name = re.split(r"\s{2,}", line.strip())[0]
        tok = line.strip()[len(name):].split()
        exact[name], ref_auc = float(tok[0]), float(tok[1])

    def auc_for(value, decimals):
        for c in auc_calls:
            if round(c["auc"], decimals) == round(value, decimals):
                return (c["auc"], c["ci"][0], c["ci"][1])
        return None

    def decimals_of(v):
        return 4 if abs(round(v, 4) - round(v, 3)) > 1e-9 else 3

    model_auc = {}
    for name, value in exact.items():
        found = auc_for(value, decimals_of(value)) if auc_calls else None
        model_auc[name] = found or next((a for t, a in aucs.items() if t.startswith(name)), None)
    ref = auc_for(ref_auc, decimals_of(ref_auc)) if auc_calls else None
    if ref is None:
        ref = next(a for t, a in aucs.items() if t.startswith("Bernoulli Naive Bayes"))
    return model_auc, tests, ref


def regenerate_table(tex, label, blocks):
    a = tex.index("\\label{" + label + "}")
    a = tex.index("\\begin{tabular}", a)
    b = tex.index("\\end{tabular}", a)
    lines = tex[a:b].split("\n")

    data = {}
    for cite, script, reference, mapping, ref_row in blocks:
        data[cite] = (block_data(script, reference), mapping, ref_row)

    # Collect raw p-values of the whole table for the BH families.
    entries = []  # (line index, cite, script model name)
    cite = None
    for i, line in enumerate(lines):
        m = re.search(r"\\cite\{(\w+)\}", line)
        if m:
            cite = m.group(1)
        if cite in data and line.rstrip().endswith("\\\\") and "&" in line:
            (model_auc, tests, ref), mapping, ref_row = data[cite]
            algo = strip(line.split("&")[-5 if label != "tab:diabetes_statistical_comparison" else -5])
            entries.append((i, cite, algo))

    family = {"DeLong": [], "McNemar": []}
    keys = []
    for i, cite, algo in entries:
        (model_auc, tests, ref), mapping, ref_row = data[cite]
        if algo == ref_row:
            continue
        name = resolve(mapping, algo, tests)
        keys.append((i, name))
        for t in family:
            family[t].append(tests[name][t][0])
    adj = {t: bh(v) for t, v in family.items()}

    for n, (i, name) in enumerate(keys):
        cite_i = [c for j, c, _ in entries if j == i][0]
        (model_auc, tests, ref), mapping, ref_row = data[cite_i]
        cells = lines[i].rstrip()[:-2].split("&")
        cells[-3] = " " + mark(cells[-3], fmt_auc(model_auc[name])) + " "
        cells[-2] = " " + fmt_p(adj["DeLong"][n], tests[name]["DeLong"][1], 3) + " "
        cells[-1] = " " + fmt_p(adj["McNemar"][n], tests[name]["McNemar"][1], 3) + " "
        lines[i] = "&".join(cells) + "\\\\"
    for i, cite_i, algo in entries:
        (model_auc, tests, ref), mapping, ref_row = data[cite_i]
        if algo == ref_row:
            cells = lines[i].rstrip()[:-2].split("&")
            cells[-3] = " " + mark(cells[-3], fmt_auc(ref)) + " "
            lines[i] = "&".join(cells) + "\\\\"
    report = {t: (int((np.array(family[t]) < 0.05).sum()), int((adj[t] < 0.05).sum())) for t in family}
    return tex[:a] + "\n".join(lines) + tex[b:], report


def resolve(mapping, algo, tests):
    if isinstance(mapping, dict):
        return mapping[algo]
    plain = algo.replace("Naïve", "Naive")
    if plain in tests:
        return plain
    raise KeyError(f"No test row for '{algo}' (available: {list(tests)})")


def regenerate_table4(tex):
    pv = json.load(open(os.path.join(PV, "table4.json"), encoding="utf-8"))
    order = ["mutual_info", "gini_gain", "information_gain", "auc_split", "otsu", "median", "mean",
             "youden_j", "kmeans_midpoint"]
    labels = {"mutual_info": "Mutual information", "gini_gain": "Gini gain", "information_gain": "Information gain",
              "auc_split": "AUC split", "otsu": "Otsu", "median": "Median", "mean": "Mean",
              "youden_j": "Youden's $J$", "kmeans_midpoint": "K-means midpoint"}
    cols = {"Diabetes": (2, 3), "Breast Cancer": (5, 6), "Heart Disease": (8, 9)}
    a = tex.index("\\label{tab:binarization_variants_three_datasets}")
    a = tex.index("\\begin{tabular}", a)
    b = tex.index("\\end{tabular}", a)
    lines = tex[a:b].split("\n")
    report = {}
    for ds, (cd, cm) in cols.items():
        for t, col in (("DeLong", cd), ("McNemar", cm)):
            raw = np.array([pv[ds][v][t] for v in order])
            adj = bh(raw)
            report[(ds, t)] = (int((raw < 0.05).sum()), int((adj < 0.05).sum()))
            for v, p in zip(order, adj):
                i = next(k for k, l in enumerate(lines) if strip(l.split("&")[0]) == labels[v])
                cells = lines[i].rstrip()[:-2].split("&")
                better = "$^{*}$" in cells[col]  # direction as in the original table
                cells[col] = " " + fmt_p(p, better, 4) + " "
                lines[i] = "&".join(cells) + "\\\\"
    return tex[:a] + "\n".join(lines) + tex[b:], report


if __name__ == "__main__":
    path = sys.argv[1]
    tex = open(path, encoding="utf-8").read()
    tex, rep = regenerate_table4(tex)
    print("Table 4 (significant raw -> BH):", rep)
    for label, blocks in BLOCKS.items():
        tex, rep = regenerate_table(tex, label, blocks)
        print(label, "(significant raw -> BH):", rep)
    open(path, "w", encoding="utf-8", newline="").write(tex)
