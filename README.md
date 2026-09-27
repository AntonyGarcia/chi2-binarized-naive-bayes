# Interpretable and Calibrated Classification of Clinical Data Using Supervised Feature Binarization

Code and data for the paper:

> Garcia, A.; Noriega de la Colina, A.; Britton, G.; Huang, X. *Interpretable and Calibrated
> Classification of Clinical Data Using Supervised Feature Binarization.* arXiv:2607.15394, 2026.
> [https://arxiv.org/abs/2607.15394](https://arxiv.org/abs/2607.15394)

The method binarizes each continuous clinical feature at a single threshold selected with the χ²
test of independence and models the result with Bernoulli Naïve Bayes (BNB). Thresholds are
estimated on the training folds only, so every reported result is free of information leakage.
This repository reproduces all analyses of the paper on three public benchmark datasets: the Pima
Indians Diabetes Database, the Wisconsin Breast Cancer (Diagnostic) dataset and the Heart Failure
Prediction dataset.

## Repository structure

```
algorithms/           The proposed model
  bnb.py              χ² threshold selection + Bernoulli NB (Categorical NB for categorical features)
  bnb_nan.py          NaN-aware Bernoulli NB (skips missing features in the likelihood)
datasets/             The three benchmark datasets (CSV)
implementations/      Re-implementations of the models of the comparison studies (one script per study;
                      see implementations/README.md for the studies and their DOIs)
  diabetes/           Table 5
  breast_cancer/      Table 6
  heart_disease/      Table 7
  */metadata/         Details of each study used for the re-implementation
method_evaluation/    Analyses of the proposed method (Tables 4, 8, 10-14 and Figure 1)
  outputs/            Results produced by the scripts, as reported in the paper
```

## Installation

Python 3.12 was used. Install the dependencies with

```bash
pip install -r requirements.txt
```

PyTorch is only needed for `implementations/diabetes/ashour_et_al.py` (feedforward and
convolutional networks). On Windows with Anaconda, PyTorch may fail to load in the base
environment; in that case run that script from a dedicated conda environment with PyTorch
installed.

## Running the analyses

Run every script from the repository root, for example

```bash
python method_evaluation/bnb_binarization_variants_three_datasets.py
```

The scripts print their results; the comparison scripts in `implementations/` can be redirected
to a file (`python implementations/heart_disease/kumar_et_al.py > kumar.txt`). Most scripts finish
in a few minutes; the neural-network and SVM comparisons (Ashour et al., Reza et al., Jakhar et
al.) can take 20-30 minutes.

| Paper item | Script | Saved output |
|---|---|---|
| Table 4 (binarization criteria) | `method_evaluation/bnb_binarization_variants_three_datasets.py` | `outputs/binarization_variants_cnb.txt` |
| Table 5 (Pima Indians Diabetes comparisons) | `implementations/diabetes/*.py` | `outputs/pvalues/<study>.txt` |
| Table 6 (Wisconsin Breast Cancer comparisons) | `implementations/breast_cancer/*.py` | `outputs/pvalues/<study>.txt` |
| Table 7 (Heart Failure comparisons) | `implementations/heart_disease/*.py` | `outputs/pvalues/<study>.txt` |
| Table 8 and Figure 1 (calibration) | `method_evaluation/calibration_analysis.py` | `outputs/calibration_table.csv`, `outputs/calibration_curves.png` |
| Tables 10-12 (thresholds and conditional probabilities) | `method_evaluation/chi2_feature_summary_tables.py` | `outputs/chi2_feature_summary_tables_v2.txt` |
| Table 14 and Table S4 (threshold stability) | `method_evaluation/threshold_stability_repeated_cv.py` | `outputs/threshold_stability*.csv` |
| Table 13 (leave-one-out worked example) | `method_evaluation/loo_diabetes_tp_breakdown.py --sample_idx 43` | Excel report in `outputs/` |

`outputs/` refers to `method_evaluation/outputs/`.

### Multiple-comparison correction (Tables 4-7)

The DeLong and McNemar p-values in Tables 4-7 are adjusted with the Benjamini-Hochberg procedure,
separately for each dataset and each test. The correction uses full-precision p-values, which are
recorded by running each comparison script through a small wrapper:

```bash
python method_evaluation/capture_test_pvalues.py implementations/heart_disease/kumar_et_al.py \
       method_evaluation/outputs/pvalues/kumar_et_al.json > method_evaluation/outputs/pvalues/kumar_et_al.txt
python method_evaluation/table4_pvalues.py
```

`method_evaluation/outputs/pvalues/` contains these records for every study. The script
`method_evaluation/regenerate_comparison_tables.py` applies the correction and writes the adjusted
values into the LaTeX tables of the manuscript.

## Evaluation protocol

- Stratified 10-fold cross-validation (seed 42); every metric is computed on the pooled
  out-of-fold predictions, with 95% bootstrap confidence intervals (1000 replicates).
- For each feature, the threshold is the candidate that maximizes the χ² statistic (with Yates'
  continuity correction) on the training fold; the test fold is binarized with that threshold.
- Class-conditional probabilities use Laplace smoothing (α = 1).
- Pima Indians Diabetes: zeros in Glucose, Blood Pressure, Skin Thickness, Insulin and BMI are
  treated as missing; the NaN-aware model omits them from the likelihood.
- Heart Failure: the five continuous predictors are binarized; the six categorical predictors keep
  their categories and are modeled with Categorical Naïve Bayes. Cholesterol values of 0 are
  replaced by the median of the non-zero values, and records with resting blood pressure below
  80 mmHg or cholesterol outside 100-450 mg/dL are removed (908 records).
- Models are compared with DeLong's test (AUC) and McNemar's test (paired predictions).

## Re-implementations of the comparison studies

Each script in `implementations/` follows the model and preprocessing description of one published
study, but all models are evaluated with the protocol above rather than with the original
validation scheme. Their AUC values are therefore not expected to match the published ones
exactly. Hyperparameters that the studies did not report were set to reasonable defaults; the
choices are documented at the top of each script and in `implementations/<dataset>/metadata/`.
The studies and their DOIs are listed in [`implementations/README.md`](implementations/README.md). In the
re-implementation of Azar and El-Said (2013), the per-sample unit-length normalization of the PNN
was omitted because it lowered the AUC on the 30-feature diagnostic dataset (see the script).

## Data

| Dataset | File | Records | Source |
|---|---|---|---|
| Pima Indians Diabetes Database | `datasets/diabetes.csv` | 768 | Smith et al. (1988), National Institute of Diabetes and Digestive and Kidney Diseases |
| Breast Cancer Wisconsin (Diagnostic) | `datasets/breast_cancer.csv` | 569 | Wolberg et al. (1993), UCI Machine Learning Repository |
| Heart Failure Prediction | `datasets/heart_disease.csv` | 918 | Fedesoriano (2021), combining UCI heart disease cohorts (Janosi et al.) |

Please cite the original sources when using the datasets.

## Citation

If you use this code, please cite the paper:

```bibtex
@misc{garcia2026interpretable,
  title         = {Interpretable and Calibrated Classification of Clinical Data Using Supervised Feature Binarization},
  author        = {Garcia, Antony and Noriega de la Colina, Adri{\'a}n and Britton, Gabrielle and Huang, Xinming},
  year          = {2026},
  eprint        = {2607.15394},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2607.15394}
}
```

The journal version is under review; this reference will be updated once it is published.
