# Financial Fraud Detection with Machine Learning

Detects fraudulent credit card transactions in a dataset where only **0.17%** of transactions are fraud
(492 of 284,807). The project focuses on doing the evaluation honestly: on data this imbalanced, a model
that predicts "legitimate" for everything scores **99.83% accuracy and catches zero fraud**, so accuracy
is not used to judge the models.

## Results (test set: 56,962 transactions, 99 frauds)

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | Frauds caught | False alarms |
|---|---|---|---|---|---|---|---|
| Always "legitimate" (baseline) | – | 0.00 | – | – | – | 0 / 99 | 0 |
| Logistic Regression | 0.886 | 0.788 | 0.834 | 0.758 | 0.966 | 78 / 99 | 10 |
| **Random Forest** | **0.931** | **0.818** | **0.871** | **0.875** | **0.977** | **81 / 99** | **6** |

The Random Forest flags 87 transactions, 81 of which are real fraud, out of 56,962. These numbers are
produced by the `train` job in GitHub Actions on the full dataset (see the latest CI run; `metrics.json`
and `pr_curve.png` are uploaded as an artifact).

## How it works (`fraud_detection.py`)

1. **Stratified 60 / 20 / 20 split** into train, validation and test sets, so each keeps the 0.17% fraud rate.
2. **Class weights** (`class_weight="balanced"`) make the rare fraud class count more during training,
   instead of resampling the data.
3. **No data leakage:** the Logistic Regression scaler is fitted on the training set only (inside a scikit-learn pipeline).
4. **Threshold tuning:** each model's decision threshold is chosen on the *validation* set to maximise F1,
   instead of the default 0.5. The *test* set is used once, for the final numbers.
   (Logistic Regression with balanced weights pushes scores very close to 1, so its tuned threshold rounds to 1.0.)
5. **Evaluation** with precision, recall, F1, PR-AUC and ROC-AUC, plus the confusion matrix. PR-AUC is the main
   metric because it ignores the huge number of easy true negatives.

## Run it

```bash
pip install -r requirements.txt
curl -L -o creditcard.csv https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
python fraud_detection.py --data creditcard.csv --out reports
```
Prints the metrics and writes `reports/metrics.json` and `reports/pr_curve.png`. Takes about a minute.

## Tests

```bash
pip install pytest
pytest -q
```
Six fast tests run on a small synthetic dataset with the same columns: stratified split, threshold
selection, confusion-matrix counts, scaler fitted on training data only, the full pipeline, and input validation.

## CI

`.github/workflows/ci.yml` runs the tests on every push, then downloads the real dataset, trains both
models and uploads the reports.

## Dataset

Credit card transactions made by European cardholders in September 2013 (Machine Learning Group, ULB;
also on Kaggle). Features `V1`–`V28` are PCA-anonymised; `Time` and `Amount` are raw; `Class` is 1 for fraud.
The CSV (~150 MB) is not committed.

## Limitations and next steps

- The split is random, not time-based; a time-based split would better mimic real deployment.
- The threshold optimises F1. A bank would pick it from the cost of a missed fraud versus a false alarm.
- 18 of 99 frauds are still missed; gradient boosting (e.g. XGBoost) is the obvious next model to try.
