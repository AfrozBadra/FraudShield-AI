# FraudShield AI

FraudShield AI is a local Streamlit dashboard for exploring a saved credit-card
fraud detection model. It is inference-only: it loads
`models/fraudshield_random_forest.joblib`, scores transactions with the saved
feature list and decision threshold, and never retrains or overwrites the
model.

## What the dashboard does

- **Overview** — saved-model snapshot, verified held-out test metrics, confusion
  matrix, and plain-language metric definitions.
- **Risk Analysis** — score a row from the local `creditcard.csv`, or enter all
  30 model features in the custom-input mode. Dataset labels are shown
  separately from model predictions.
- **Dataset Explorer** — row and column counts, class distribution, fraud rate,
  missing values, duplicate rows, schema, and a small preview.
- **Model Performance** — fixed held-out test results, error analysis, threshold
  policy, and caveats about the small number of test fraud cases.
- **Model Details** — artifact contents, evaluation setup, input contract,
  runbook, and limitations.

## Model and decision threshold

The saved artifact contains a scikit-learn `RandomForestClassifier` configured
with:

- `n_estimators=100`
- `class_weight="balanced"`
- `random_state=42`
- 30 input features, in this order: `Time`, `V1`–`V28`, `Amount`

The artifact also stores the decision threshold (`0.20`). Live predictions use
that artifact threshold, and the fraud probability is read from the
`predict_proba` column that corresponds to class label `1`. The dashboard
validates that the model exposes `classes_` ordered as `[0, 1]` and that its
`feature_names_in_` (when available) exactly matches the saved feature list.

The historical evaluation metrics below were produced at threshold `0.20`. If
an artifact with a different threshold is loaded, the dashboard explains the
difference and continues to show the historical metrics unchanged; it does not
recalculate them.

## Verified test results

Held-out test-set results at decision threshold `0.20`:

| Metric | Value |
| --- | --- |
| Precision | 82.61% |
| Recall | 80.28% |
| F1-score | 0.8143 |
| Average Precision | 0.7967 |
| True negatives | 42,476 |
| False positives | 12 |
| False negatives | 14 |
| True positives | 57 |
| Test transactions | 42,559 |
| Actual fraud cases | 71 |

These are fixed results from `07_final_evaluation.py`; the dashboard does not
recompute them from test labels.

## Dataset

**Original source:**
[Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

FraudShield AI uses the Credit Card Fraud Detection dataset. The original
dataset contains **284,807 transactions and 31 columns**, including the target
column `Class`:

- `Class = 1` indicates a fraudulent transaction.
- `Class = 0` indicates a legitimate transaction.

The original data contains anonymized features `V1`–`V28`, plus `Time` (seconds
elapsed since the first record) and `Amount` (transaction amount). `V1`–`V28`
are principal components produced by PCA, not raw business fields.

### Attribution

The Kaggle dataset page lists the publisher as **Machine Learning Group - ULB**
(Université Libre de Bruxelles), with the data collected and analysed in a
research collaboration with **Worldline**. The page's acknowledgements credit
Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi, and
ask users to cite:

> Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi.
> Calibrating Probability with Undersampling for Unbalanced Classification.
> In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015.

This project does not claim ownership of the original dataset. The repository
does not redistribute the CSV.

### License status

The Kaggle dataset page labels the dataset **“Database: Open Database, Contents:
Database Contents”** and links to the Open Data Commons Database Contents License
(DbCL) 1.0. The exact terms should be confirmed on the
[source page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) before
redistributing or republishing the data. This project does not include a
separate license file for the dataset.

### Original data versus the modeling data

The original dataset has **284,807 rows**. This project's modeling workflow
removes exact duplicate rows before training (`02_data_preparation.py`,
`06_threshold_tuning.py`, `07_final_evaluation.py`, and `08_save_model.py` all
call `drop_duplicates()`), so the processed row count differs from the original
size. For the local copy used here, 1,081 duplicate rows are reported, leaving
283,726 rows for modeling. The dashboard reads the local CSV as-is and displays
its duplicate count; it does not silently rewrite the file.

Preprocessing performed by the project scripts:

1. Read `creditcard.csv`.
2. Remove exact duplicate rows.
3. Use every column except `Class` as the 30-feature input matrix.
4. Create a stratified 70% train / 15% validation / 15% test split with
   `random_state=42`.
5. Train the Random Forest on the training split and select the threshold using
   validation-set F1. The held-out test split is used only for final evaluation.

## Installation

The project was verified in a Python 3.13 environment with the versions pinned
in `requirements.txt`:

- `streamlit==1.66.0`
- `pandas==3.0.6`
- `numpy==2.5.3`
- `scikit-learn==1.9.1`
- `joblib==1.6.0`

From the project folder (the folder containing `10_app.py`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```bat
.venv\Scripts\activate
```

No Kaggle API token, account, or network access is required to view the
dashboard. The app only reads local files.

## Obtain `creditcard.csv`

1. Open the dataset page:
   <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>.
2. Download the dataset from Kaggle. The archive contains `creditcard.csv`
   (about 144 MiB / 150,828,752 bytes for the local copy used here).
3. Place the file directly in this project folder, next to `10_app.py`:

   ```text
   FraudShield AI/
   ├── 10_app.py
   ├── creditcard.csv
   └── models/
       └── fraudshield_random_forest.joblib
   ```

4. Confirm the file name is exactly `creditcard.csv`. The application expects
   it at `creditcard.csv` relative to `10_app.py`; it does not download the
   dataset automatically.

The CSV is listed in `.gitignore` because it exceeds GitHub's standard 100 MB
per-file limit. Keep it locally; do not commit it.

## Run the dashboard

```bash
python -m streamlit run 10_app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`).

If the saved model artifact is missing, recreate it once with:

```bash
python 08_save_model.py
```

You can verify the saved artifact separately with:

```bash
python 09_test_saved_model.py
```

The dashboard itself never trains a replacement model.

## Tests

Run the automated test suite from the project folder:

```bash
python -B -m unittest discover -s tests -v
```

The tests use Streamlit's `AppTest` and temporary directories. They do not train
a model or modify the real artifact or dataset.

## Project structure

```text
FraudShield AI/
├── 01_data_inspection.py        # Dataset inspection
├── 02_data_preparation.py       # Deduplication and 70/15/15 split
├── 03_baseline_model.py         # Baseline experiment
├── 04_logistic_regression.py    # Logistic regression experiment
├── 05_model_comparison.py       # Model comparison
├── 06_threshold_tuning.py       # Validation-set threshold tuning
├── 07_final_evaluation.py       # Final held-out test evaluation
├── 08_save_model.py             # Artifact creation
├── 09_test_saved_model.py       # Saved-artifact smoke check
├── 10_app.py                    # Streamlit dashboard
├── creditcard.csv               # Local dataset (not committed)
├── models/
│   └── fraudshield_random_forest.joblib
├── tests/
│   └── test_fraudshield_app.py
├── .streamlit/config.toml
├── .gitignore
├── requirements.txt
└── README.md
```

## Deployment

No public FraudShield AI deployment URL is currently confirmed. The previously
known `machineguard-aiml.streamlit.app` address belongs to a different
(MachineGuard) project and is not a FraudShield AI deployment. Run the app
locally using the instructions above.

## Limitations

- This is an **educational ML demonstration**, not a production-grade bank
  fraud detection service.
- The model was trained on one historical, anonymized dataset. It may not
  transfer to another bank, region, time period, or feature pipeline.
- The held-out test set contains only 71 fraud cases, so a single case moves
  recall by roughly 1.4 percentage points. Small metric differences are
  uncertain.
- Overall accuracy hides the class imbalance and is not the headline metric
  here.
- A model probability is an estimate, not a guarantee. Any real risk process
  needs human review and appropriate controls.
- The dashboard uses no live transactions, banking connections, external APIs,
  or telemetry. It reads local files only.
- Avoid upgrading scikit-learn without re-exporting the joblib artifact, because
  saved model artifacts can be version-sensitive.
