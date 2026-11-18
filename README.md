# An Explainable Ensemble Machine Learning Model for Detecting Transaction-Level Fraud Patterns in Mobile Money Applications

Final-year project, BSc Computer Networks and Cybersecurity, Strathmore University (2026).
Supervisor: Mr Harrison Talo.

Phishing and social engineering attacks such as smishing, vishing, SIM-swap fraud and impersonation scams are invisible in transaction data. This project does not detect the attack itself. It detects the transactional footprint the attack leaves behind: large sums moved to new or dormant recipients at unusual hours.

A soft-voting ensemble of Random Forest and XGBoost scores each transaction, SHAP explains every flagged decision, and a real-time prototype puts the model in front of fraud analysts: a FastAPI decision service with step-up verification and a tamper-evident audit log, and the Mlinzi analyst dashboard.

## Results (test set)

| Metric | Value |
| --- | --- |
| PR-AUC (primary metric) | 0.6575 (95% CI 0.633 to 0.683); random baseline 0.033 |
| ROC-AUC | 0.914 |
| Alert precision | 0.873 (almost nine in ten alerts are genuine fraud) |
| Alert recall | 0.423 |
| False-positive rate | 0.0021 |
| Decision latency | 29.1 ms median, 38.7 ms 95th percentile |
| Functional tests | 18 of 18 passed |

## How it works

1. **Data:** the PaySim synthetic mobile money dataset. Fraud occurs only in TRANSFER and CASH_OUT, so modelling uses those 2,770,409 transactions (0.30% fraud), split chronologically into training, validation and test periods so that every model is tested only on transactions that happened after the ones it learned from.
2. **Features:** 13 features were engineered so that none uses information unavailable at the moment of authorization. The final model uses 10 of them: amount, log amount, transaction type, hour of day and six recipient-history features.
3. **Model selection:** nine candidates were compared, including Logistic Regression, a Decision Tree, Isolation Forest, Random Forest, XGBoost and an Isolation Forest + XGBoost hybrid. The soft-voting ensemble was the only challenger that beat XGBoost on all five random seeds; the originally proposed hybrid was rejected.
4. **Calibration and risk bands:** Platt scaling turns scores into probabilities, which map to four bands: Low (approve), Medium (step-up verification), High (hold and alert) and Critical (decline and escalate).
5. **Explainability:** SHAP TreeExplainer attributions, averaged across both ensemble members, show which features drove each decision. The recipient's history contributes most, followed by the amount, the transaction type and the hour.
6. **Prototype:** the decision API scores a transaction, assigns a band and action, records the decision in a hash-chained audit log and returns it; the Mlinzi dashboard gives analysts an alert queue, explanations, case management and live replay.

### Why the balance features were excluded

Adding three sender-balance features lifted XGBoost to a PR-AUC of 0.994. That score comes from how PaySim generates fraud (98.1% of fraudulent transactions move exactly the whole balance), not from a better model, so those features were excluded. The post-transaction balance fields and the simulator's `isFlaggedFraud` flag were also excluded because they encode the outcome.

### Limitations

- PaySim is simulated, so performance on real mobile money data cannot be assumed.
- The model covers TRANSFER and CASH_OUT only. PAYMENT transactions carry no fraud labels in PaySim, so merchant, till and paybill fraud is outside its coverage and is recorded as not assessed.
- There is no device, location or SIM-swap data, so the system detects the transactional consequences of social engineering rather than the attack itself.

## Project structure

```
src/        Pipeline, model, API and dashboard code
tests/      Tests
results/    Figures, metrics and SHAP outputs behind the reported results
models/     Model settings (thresholds, risk and rule configs); trained .joblib files are not committed
assets/     Logo and icon
.streamlit/ Dashboard theme
```

| Stage | Scripts in `src/` |
| --- | --- |
| Data preparation | `prepare_data.py`, `split_data.py` |
| Features | `features.py`, `build_features_v2.py`, `feature_store.py` |
| Training | `train_models.py`, `train_ensemble.py`, `run_experiments.py`, `calibrate_final.py` |
| Evaluation | `evaluate_models.py`, `evaluate_rules.py`, `seed_robustness.py`, `ensemble_seed_check.py`, `benchmark_latency.py` |
| Explainability | `explain.py` |
| Decision service | `api.py`, `decision_engine.py`, `rules.py`, `audit_log.py` |
| Analyst dashboard | `dashboard.py` |

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
