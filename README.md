# Explainable Ensemble Machine Learning for Mobile Money Fraud Detection

Final-year project, BSc Computer Networks and Cyber Security, Strathmore University (2026).
Supervisor: Mr Harrison Talo.

This project detects transactional- fraud in mobile money transfer applications. A Random Forest + XGBoost soft-voting ensemble scores each transaction, SHAP explains every flagged decision, and a FastAPI service with a Streamlit dashboard puts the model in front of an analyst in real time.

## Results


| Metric              | Value                                   |
| ------------------- | --------------------------------------- |
| Test PR-AUC         | 0.658 (baseline 0.033)                  |
| Precision           | 0.873 (87% of alerts are genuine fraud) |
| Recall              | 0.423                                   |
| False-positive rate | 0.2%                                    |
| API latency         | 29 ms median                            |
| Functional tests    | 18 of 18 passed                         |

## How it works

1. **Data:** the PaySim synthetic mobile money dataset, filtered to 2.77 million TRANSFER and CASH_OUT transactions (0.30% fraud), split by time so the model never trains on the future.
2. **Features:** 13 behavioural features, including recipient history, transaction amount and hour of day.
3. **Model selection:** nine models compared; the Random Forest + XGBoost soft-voting ensemble beat XGBoost alone on all five random seeds.
4. **Explainability:** SHAP values show which features drove each fraud alert.
5. **Deployment:** a FastAPI decision service with step-up verification and a tamper-evident audit log, plus a Streamlit analyst dashboard.

### A note on leakage

PaySim's four balance columns (`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`) pushed PR-AUC to 0.994 because of how the simulator handles fraudulent transactions. That score is unrealistic, so those features were excluded and the reported results use behavioural features only.

## Project structure

```
src/        Pipeline, model, API and dashboard code
tests/      Tests
results/    Figures, metrics and SHAP outputs behind the reported results
models/     Model settings (thresholds, risk and rule configs); trained .joblib files are not committed
assets/     Logo and icon
.streamlit/ Dashboard theme
```


| Stage             | Scripts in`src/`                                                                                                  |
| ----------------- | ----------------------------------------------------------------------------------------------------------------- |
| Data preparation  | `prepare_data.py`, `split_data.py`                                                                                |
| Features          | `features.py`, `build_features_v2.py`, `feature_store.py`                                                         |
| Training          | `train_models.py`, `train_ensemble.py`, `run_experiments.py`, `calibrate_final.py`                                |
| Evaluation        | `evaluate_models.py`, `evaluate_rules.py`, `seed_robustness.py`, `ensemble_seed_check.py`, `benchmark_latency.py` |
| Explainability    | `explain.py`                                                                                                      |
| Decision service  | `api.py`, `decision_engine.py`, `rules.py`, `audit_log.py`                                                        |
| Analyst dashboard | `dashboard.py`                                                                                                    |

## Getting started

### 1. Get the data

The dataset is too large for GitHub, so it is not included. Download it from Kaggle:
[Synthetic Financial Datasets For Fraud Detection (PaySim)](https://www.kaggle.com/datasets/ealaxi/paysim1)

Save the CSV as `Data/raw/PS_20174392719_1491204439457_log.csv`.

### 2. Install

Developed with Python 3.14.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file in the project root for local settings (it is never committed).

### 3. Run

Run the data preparation, feature and training scripts in the order of the table above, then:

```bash
# Start the decision API
uvicorn src.api:app --reload

# Start the analyst dashboard
streamlit run src/dashboard.py
```

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · SHAP · FastAPI · SQLite · Streamlit · Matplotlib

## Author

Adrian Rutto · [LinkedIn](https://www.linkedin.com/in/adrian-rutto-314984332/)
