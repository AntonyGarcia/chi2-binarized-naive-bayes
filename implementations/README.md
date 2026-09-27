# Re-implemented comparison studies

Each script re-implements the models of one published study on the same dataset and compares them
with the proposed Bernoulli Naïve Bayes model (DeLong and McNemar tests). All models are evaluated
with the protocol of this repository (stratified 10-fold cross-validation, pooled out-of-fold
predictions, seed 42), not with the original validation scheme, so the AUC values are not expected
to match the published ones exactly. The `metadata/` folder of each dataset summarizes the details
of each study used for the re-implementation (preprocessing, hyperparameters, reported results).
The articles themselves are not included in this repository.

## Pima Indians Diabetes Database (Table 5)

| Script | Study | Models | DOI |
|---|---|---|---|
| `diabetes/ashour_et_al.py` | Ashour, Fouda, Fadlullah and Ibrahem (2024). Optimized neural networks for diabetes classification using Pima Indians Diabetes Database. *IEEE ICMI 2024* | Gaussian NB, k-NN, J48, random forest, feedforward NN, CNN | [10.1109/ICMI60790.2024.10585703](https://doi.org/10.1109/ICMI60790.2024.10585703) |
| `diabetes/battineni_et_al.py` | Battineni et al. (2019). Comparative machine-learning approach: a follow-up study on type 2 diabetes predictions by cross-validation methods. *Machines* 7, 74 | Gaussian NB, J48, random forest, logistic regression | [10.3390/machines7040074](https://doi.org/10.3390/machines7040074) |
| `diabetes/bhoi_et_al.py` | Bhoi et al. (2021). Prediction of diabetes in females of Pima Indian heritage: a complete supervised learning approach. *Turkish Journal of Computer and Mathematics Education* 12, 3074–3084 | Classification tree, SVM, k-NN, Gaussian NB, random forest, MLP, AdaBoost, logistic regression | No DOI registered |
| `diabetes/pranto_et_al.py` | Pranto et al. (2020). Evaluating machine learning methods for predicting diabetes among female patients in Bangladesh. *Information* 11, 374 | Gaussian NB, decision tree, random forest, k-NN | [10.3390/info11080374](https://doi.org/10.3390/info11080374) |
| `diabetes/reza_et_al.py` | Reza, Hafsha, Amin, Yasmin and Ruhi (2023). Improving SVM performance for type II diabetes prediction with an improved non-linear kernel: insights from the PIMA dataset. *Computer Methods and Programs in Biomedicine Update* 4, 100118 | SVM with linear, polynomial and distance-based RBF kernels | [10.1016/j.cmpbup.2023.100118](https://doi.org/10.1016/j.cmpbup.2023.100118) |

## Wisconsin Breast Cancer, Diagnostic (Table 6)

| Script | Study | Models | DOI |
|---|---|---|---|
| `breast_cancer/azar_et_al.py` | Azar and El-Said (2013). Probabilistic neural network for breast cancer classification. *Neural Computing and Applications* 23, 1737–1751 | MLP, RBF network, probabilistic NN | [10.1007/s00521-012-1134-8](https://doi.org/10.1007/s00521-012-1134-8) |
| `breast_cancer/jakhar_et_al.py` | Jakhar, Gupta and Singh (2023). SELF: a stacked-based ensemble learning framework for breast cancer classification. *Evolutionary Intelligence* 17, 1341–1356 | SELF stacking, extra trees, random forest, AdaBoost, gradient boosting, k-NN, SVM, MLP, CART, SGD | [10.1007/s12065-023-00824-4](https://doi.org/10.1007/s12065-023-00824-4) |
| `breast_cancer/naji_et_al.py` | Naji et al. (2021). Machine learning algorithms for breast cancer prediction and diagnosis. *Procedia Computer Science* 191, 487–492 | SVM, random forest, logistic regression, decision tree, k-NN | [10.1016/j.procs.2021.07.062](https://doi.org/10.1016/j.procs.2021.07.062) |
| `breast_cancer/roshenov.py` | Rovshenov and Peker (2022). Performance comparison of different machine learning techniques for early prediction of breast cancer using Wisconsin Breast Cancer Dataset. *IISEC 2022* | ANN, SVM (RBF), random forest | [10.1109/IISEC56263.2022.9998248](https://doi.org/10.1109/IISEC56263.2022.9998248) |
| `breast_cancer/shah_et_al.py` | Shah et al. (2022). Machine learning techniques for identification of carcinogenic mutations, which cause breast adenocarcinoma. *Scientific Reports* 12, 11738 | Decision tree, Gaussian NB, random forest | [10.1038/s41598-022-15533-8](https://doi.org/10.1038/s41598-022-15533-8) |

## Heart Failure Prediction (Table 7)

| Script | Study | Models | DOI |
|---|---|---|---|
| `heart_disease/kumar_et_al.py` | Kumar Mygapula, Lal Saini and Raj Dheeraj (2023). Performance evaluation of machine learning algorithms for prediction of cardiac failure. *ICSCNA 2023* | SVM, random forest, Naïve Bayes, logistic regression, k-NN, decision tree | [10.1109/ICSCNA58489.2023.10368606](https://doi.org/10.1109/ICSCNA58489.2023.10368606) |
| `heart_disease/liu_et_al.py` | Liu, Dong, Zhao and Tian (2022). Predictive classifier for cardiovascular disease based on stacking model fusion. *Processes* 10, 749 | Logistic regression, random forest, extra trees, GBDT, MLP, stacking | [10.3390/pr10040749](https://doi.org/10.3390/pr10040749) |
| `heart_disease/muhammad_et_al.py` | Muhammad et al. (2023). Enhancing prognosis accuracy for ischemic cardiovascular disease using K nearest neighbor algorithm: a robust approach. *IEEE Access* 11, 97879–97895 | Logistic regression, decision tree, random forest, SVM (RBF), k-NN, Gaussian NB | [10.1109/ACCESS.2023.3312046](https://doi.org/10.1109/ACCESS.2023.3312046) |
| `heart_disease/patidar.py` | Patidar, Jain and Gupta (2022). Comparative analysis of machine learning algorithms for heart disease predictions. *ICICCS 2022* | Random forest, logistic regression, k-NN | [10.1109/ICICCS53718.2022.9788408](https://doi.org/10.1109/ICICCS53718.2022.9788408) |

## Notes

- Hyperparameters that a study did not report were set to reasonable defaults; the choices are
  documented at the top of each script and in `metadata/`.
- `breast_cancer/azar_et_al.py`: the study used the 9-feature Wisconsin (Original) dataset; here the
  models are evaluated on the 30-feature diagnostic dataset. The per-sample unit-length
  normalization of the probabilistic NN is omitted because it lowered the AUC on this dataset.
- `diabetes/ashour_et_al.py` requires PyTorch.
