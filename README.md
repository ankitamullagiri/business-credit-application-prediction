# Business Credit Application Prediction

Predicting which bank customers are likely to apply for business credit, so relationship managers can reach them *before* they go to a competitor. The core challenge is a 9:1 class imbalance.

**Context:** MSc Business Analytics, Warwick Business School (Analytics in Practice) · group project · Python

📊 [Presentation slides](docs/slides.pdf)

## Business problem

- The bank only engages customers after they apply, which is reactive.
- There's no early signal of upcoming credit needs, and high-value customers drift to competitors.
- **Goal:** score every customer on their likelihood to apply, and hand managers a prioritised list.

## Data

200 anonymised numeric features (`var_0` to `var_199`) per customer and a binary `target`. Only about **10% are positive**.

## Approach

1. **Cleaning:** dropped columns with more than 30% missing and mean-imputed the rest.
2. **Stratified 70/30 split**, so both sets keep the 9:1 ratio.
3. **Feature selection:** kept the 153 features with positive mutual information with the target.
4. **Imbalance handling:** class weighting (`class_weight='balanced'`, and `scale_pos_weight ≈ 9` for XGBoost) instead of resampling.
5. **Tuning:** hyperparameters (e.g. `C` for LR and SVM) were tuned on a smaller sample (`small_dataset_tuning.ipynb`), then applied to the full data (`full_dataset_class_weighted.ipynb`).
6. **Thresholds:** decision thresholds from 0.1 to 0.5 were compared for each model to trade precision against recall.
7. **Evaluation:** ROC/AUC, precision–recall curves and cumulative gain (lift) charts.

## Results (30,000-customer test set)

| Model | Threshold | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.4 | 0.77 | 0.27 | 0.75 | 0.39 | 0.844 |
| **SVM (RBF)** | 0.2 | **0.87** | **0.42** | 0.64 | **0.51** | **0.868** |
| Random Forest | 0.1 | 0.63 | 0.19 | **0.83** | 0.31 | 0.799 |
| XGBoost | 0.5 | 0.85 | 0.37 | 0.67 | 0.48 | 0.864 |

**SVM was selected:** it has the best F1 and AUC, and finds about two-thirds of future applicants while keeping precision at roughly 4× the base rate. Random Forest is the option if missing an applicant costs far more than a wasted call.

## Limitations

- Thresholds were compared on the test set, so a separate validation set would give a cleaner final estimate.
- Random Forest overfits (training accuracy 1.0). Limiting depth or leaf size would likely improve it.

## Run it

```bash
pip install pandas numpy scikit-learn xgboost matplotlib kds
jupyter notebook notebooks/full_dataset_class_weighted.ipynb
```

The notebooks expect `datafile_full.csv` / `datafile_small.csv` in the working directory. *The data files aren't included in this repo.*

## Tech stack

Python · pandas · scikit-learn · XGBoost · imbalanced classification · threshold tuning · lift analysis
