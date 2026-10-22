# FraudShield AI — Complete ML Workflow Notes

> **Project:** Credit Card Fraud Detection  
> **Application:** Streamlit dashboard  
> **Model:** Random Forest Classifier  
> **Decision threshold:** 0.20  
> **Live app:** https://payshield-ml.streamlit.app/  
> **GitHub:** https://github.com/AfrozBadra/FraudShield-AI

> **Important:** This document records the workflow and previously reported results. The full Website, ML Logic & Deployment Audit is still in progress, so treat the metrics and implementation details as pending final verification. This is an educational prototype, not a production banking security system.

## Contents

1. [Project overview](#1-project-overview)
2. [Complete workflow](#2-complete-workflow)
3. [Problem definition](#3-step-1--define-the-problem)
4. [Dataset and EDA](#4-step-2--load-and-understand-the-dataset)
5. [Data cleaning](#5-step-3--clean-and-prepare-the-data)
6. [Features and target](#6-step-4--separate-features-and-target)
7. [Train/validation/test split](#7-step-5--split-the-data)
8. [Class imbalance](#8-step-6--understand-class-imbalance)
9. [Baseline and model comparison](#9-step-7--train-and-compare-models)
10. [Threshold selection](#10-step-8--choose-a-classification-threshold)
11. [Evaluation](#11-step-9--evaluate-the-model)
12. [Saving the model](#12-step-10--save-and-reload-the-model)
13. [Streamlit dashboard](#13-step-11--build-the-streamlit-application)
14. [Sample dataset fallback](#14-step-12--handle-the-missing-full-dataset)
15. [Testing and deployment](#15-step-13--test-and-deploy)
16. [Prediction pipeline](#16-how-the-website-makes-a-prediction)
17. [Metric interpretation](#17-how-to-interpret-the-metrics)
18. [Limitations](#18-limitations-and-responsible-claims)
19. [Interview explanation](#19-interview-ready-explanation)
20. [Interview questions](#20-interview-questions-and-answers)
21. [ML Engineer lessons](#21-ml-engineer-lessons)
22. [Final checklist](#22-final-checklist)

---

## 1. Project Overview

FraudShield AI is an end-to-end supervised machine-learning project that estimates whether a credit-card transaction may be fraudulent. A saved Random Forest model is integrated into an interactive Streamlit application.

This project covers the ML lifecycle: data inspection, preparation, class-imbalance analysis, model comparison, threshold selection, evaluation, model persistence, inference logic, testing, documentation, and deployment.

| Item | Description |
|---|---|
| Problem type | Binary classification |
| Dataset | Credit Card Fraud Detection dataset published on Kaggle by the Machine Learning Group (ULB), in collaboration with Worldline |
| Original dataset size | 284,807 transactions |
| Original columns | 31 |
| Model inputs | 30 features |
| Target | `Class` |
| Classes | `0 = legitimate`, `1 = fraud` |
| Selected model | Random Forest Classifier |
| Saved artifact | `models/fraudshield_random_forest.joblib` |
| Threshold | `0.20` |
| Web framework | Streamlit |
| Live app | https://payshield-ml.streamlit.app/ |
| Repository | https://github.com/AfrozBadra/FraudShield-AI |

### Goal

Given transaction features `X`, estimate a fraud probability and convert it into a class prediction:

```python
predicted_class = int(fraud_probability >= 0.20)
```

A model probability is an estimate, not proof that a transaction is fraudulent. This educational demo is not connected to a bank or payment processor.

---

## 2. Complete Workflow

```text
Problem definition
       |
       v
Load and inspect dataset
       |
       v
Exploratory Data Analysis (EDA)
       |
       v
Clean and prepare data
       |
       v
Separate features (X) and target (y)
       |
       v
Train / validation / test split
       |
       v
Understand class imbalance
       |
       v
Train baseline and compare models
       |
       v
Select model and threshold
       |
       v
Evaluate on held-out test data
       |
       v
Save model artifact
       |
       v
Build Streamlit inference app
       |
       v
Test, document, deploy, and audit
```

Each stage has a purpose. EDA tells us what the data contains; training fits a model; validation supports decisions; testing estimates performance; deployment makes the model usable; auditing checks that the code and claims are trustworthy.

---

## 3. Step 1 — Define the Problem

This is a **binary classification** task:

- `Class = 0`: legitimate transaction.
- `Class = 1`: fraudulent transaction.

We need to detect fraud while understanding the trade-off between false alarms and missed fraud.

**Why this matters:** The problem definition determines the target, input features, appropriate algorithms, evaluation metrics, and how a prediction should be interpreted. Because fraud is rare, accuracy alone is not a sufficient measure.

---

## 4. Step 2 — Load and Understand the Dataset

The original dataset contains:

- 284,807 transactions.
- 30 candidate model features.
- One target column, `Class`.
- 492 fraud cases in the original dataset.

| Column(s) | Meaning |
|---|---|
| `Time` | Seconds elapsed from the dataset's first transaction |
| `V1`–`V28` | Anonymized numerical features; their original business meanings are not provided |
| `Amount` | Transaction amount |
| `Class` | Ground-truth label: 0 for legitimate, 1 for fraud |

### Why inspect the data first?

We need to understand the schema, data types, target distribution, missing values, duplicates, and value ranges before training. Otherwise, data-quality problems may affect training or break inference later.

Typical inspection code:

```python
df.shape
df.head()
df.info()
df.isna().sum()
df["Class"].value_counts()
df["Class"].value_counts(normalize=True) * 100
df.duplicated().sum()
df.describe()
```

Correlation and visualizations can reveal associations, but correlation does not prove causation. The anonymized `V` features also limit business-level interpretation.

---

## 5. Step 3 — Clean and Prepare the Data

The recorded workflow identified 1,081 duplicate rows and reported 283,726 rows remaining after deduplication.

### Why consider duplicates?

Duplicate records can affect evaluation if identical examples appear in both training and test data. Removing duplicates may help, but the correct approach depends on why the duplicates exist and how the data is split.

The audit should verify:
- When duplicates are removed relative to the split.
- Whether cleaning is applied consistently.
- Whether the target distribution changes after cleaning.
- Whether any preprocessing is fitted using validation or test data.

**Important:** Do not assume that deduplication automatically improves every model. Document and evaluate the choice.

---

## 6. Step 4 — Separate Features and Target

The intended feature schema contains exactly 30 columns in this order:

```python
FEATURES = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]

X = df[FEATURES]
y = df["Class"]
```

- `X` contains information available to make a prediction.
- `y` contains the correct outcome the model learns to predict.

### Why exclude `Class` from `X`?

`Class` is the answer we want the model to predict. Including it as an input would cause target leakage, making evaluation meaningless.

### Why feature order matters

The training and inference stages must agree on feature names, order, types, and preprocessing. If columns are reordered or interpreted incorrectly, the app can produce invalid predictions even if it does not crash.

---

## 7. Step 5 — Split the Data

The recorded split was:

| Subset | Rows | Purpose |
|---|---:|---|
| Training | 198,608 | Fit the model |
| Validation | 42,559 | Compare models and choose settings/threshold |
| Test | 42,559 | Final evaluation |

These counts add to the recorded 283,726 cleaned rows.

- **Training:** used to fit model parameters.
- **Validation:** used for model selection and threshold tuning.
- **Test:** reserved for final evaluation after decisions are made.

Stratification helps preserve the rare fraud proportion in each subset. A fixed random seed supports reproducibility.

**Leakage caution:** If the same test set was repeatedly used to choose models or thresholds, the final score may be optimistic. The audit should verify the split and experiment history rather than trusting the name “test set” alone.

---

## 8. Step 6 — Understand Class Imbalance

The original dataset has 492 fraud cases among 284,807 transactions, approximately **0.173% fraud**.

A model that predicts every transaction as legitimate could achieve high accuracy while detecting no fraud. This is why accuracy can be misleading.

Important metrics:

- **Precision:** Of transactions predicted as fraud, what fraction were actually fraud?
- **Recall:** Of all actual fraud cases, what fraction did the model detect?
- **F1 score:** Harmonic mean of precision and recall.
- **Average Precision (AP):** Summarizes precision-recall ranking performance across score thresholds.
- **Confusion matrix:** Shows true positives, false positives, true negatives, and false negatives.

Oversampling, undersampling, or class weighting are possible strategies, but any resampling should be applied only to training data. Validation and test data should retain a realistic distribution.

---

## 9. Step 7 — Train and Compare Models

We explored classification models including Logistic Regression, Random Forest, and Gradient Boosting.

### Logistic Regression

A relatively simple baseline that can estimate class probabilities. It provides a reference point for more complex models.

### Random Forest

An ensemble of decision trees that can learn nonlinear patterns and feature interactions. It combines the trees' outputs to make a prediction.

### Gradient Boosting

Builds trees sequentially, with later trees attempting to correct patterns missed by earlier ones.

### Why compare models?

No algorithm is automatically best for every dataset. We compare candidates using appropriate metrics and select based on evidence rather than popularity.

The saved model in this project is a Random Forest artifact: `models/fraudshield_random_forest.joblib`.

The audit should confirm that the artifact corresponds to the final intended training process and that model selection was performed without repeatedly tuning against the test set.

### Recorded baseline

The historical Logistic Regression baseline results were reported as:

| Metric | Recorded result |
|---|---:|
| Accuracy | 82.45% |
| Precision | 14.2% |
| Recall | 82.4% |
| F1 score | 0.242 |
| Average Precision | 0.382 |
| False positives | 339 |
| False negatives | 12 |

These are historical development results and must be verified against the actual scripts before being presented as final. They illustrate that a model can catch many fraud cases while still generating many false alarms.

---

## 10. Step 8 — Choose a Classification Threshold

A classifier can estimate a fraud probability. The threshold converts the score into a class decision.

The deployed app's recorded threshold is **0.20**:

```python
predicted_class = int(fraud_probability >= 0.20)
```

For example, if the model estimates a fraud probability of 0.32:
- At threshold 0.50, the prediction would be legitimate.
- At threshold 0.20, the prediction would be fraud.

### Lower threshold

Usually catches more fraud cases, but also tends to produce more false alarms.

### Higher threshold

Usually produces fewer false alarms, but can miss more fraud cases.

### Why tune using validation data?

The threshold is a decision setting, not a model-training parameter. Choose it using validation results and the intended cost of errors—not by adjusting it until the test score looks best.

The audit must confirm the exact threshold-selection method and ensure the model and dashboard use the same threshold. Also, a predicted score is not necessarily a calibrated real-world probability.

---

## 11. Step 9 — Evaluate the Model

Previously recorded test results at threshold 0.20:

| Metric | Recorded result |
|---|---:|
| Precision | 0.8261 (82.61%) |
| Recall | 0.8028 (80.28%) |
| F1 score | 0.8143 |
| Average Precision | 0.7967 |

Previously recorded confusion matrix:

```text
TN = 42,476     FP = 12
FN = 14         TP = 57
```

These counts imply 42,559 test transactions, including 71 fraud cases, and are internally consistent with the recorded precision and recall to rounding. The audit must still reproduce them from the correct saved artifact and evaluation data.

### Precision

```text
Precision = TP / (TP + FP)
          = 57 / (57 + 12)
          ≈ 82.61%
```

About 82.61% of the predicted fraud alerts were correct on this recorded test set.

### Recall

```text
Recall = TP / (TP + FN)
       = 57 / (57 + 14)
       ≈ 80.28%
```

About 80.28% of the actual fraud cases were detected in this recorded test set.

### F1 score

The F1 score combines precision and recall into one measure. It does not directly account for the different financial costs of false positives and false negatives.

### Average Precision

Average Precision summarizes how well fraud examples are ranked ahead of legitimate examples across thresholds. It is not the percentage of individual predictions that are correct.

### Confusion matrix interpretation

- **42,476 true negatives:** legitimate transactions correctly classified.
- **12 false positives:** legitimate transactions incorrectly flagged.
- **14 false negatives:** fraud transactions missed.
- **57 true positives:** fraud transactions correctly detected.

**Evaluation caution:** These results are previously recorded, not independently reverified. The audit should confirm whether the test data influenced earlier model-selection decisions and whether the final artifact reproduces the scores. If the test set was used repeatedly for selection, state that limitation honestly.

---

## 12. Step 10 — Save and Reload the Model

The fitted Random Forest was saved with joblib:

`models/fraudshield_random_forest.joblib`

### Why save the model?

Training is an experiment; deployment needs a reusable artifact. Saving the fitted model lets the app make predictions without retraining every time.

Verify that these remain consistent:
- Model artifact and version.
- Feature names and exact order.
- Any required preprocessing.
- Fraud class identification.
- Decision threshold.
- Compatible library versions.

Testing the saved artifact is important because a model that works during training may fail when loaded in a different directory or cloud environment.

**Security note:** Only load model artifacts from trusted sources. Pickle-based formats such as joblib can execute code during deserialization.

---

## 13. Step 11 — Build the Streamlit Application

The deployed dashboard has five main areas.

### 1. Overview

Introduces the project and displays evaluation information. The UI should clearly identify metrics as offline test results, not a guarantee of future performance.

### 2. Risk Analysis

Users can select a sample transaction or enter a custom vector of 30 features. The app shows a predicted class and fraud probability.

The model prediction and the dataset's `Class` label must remain separate. The historical label is only for comparison and must not be included in model inputs.

### 3. Dataset Explorer

Displays statistics and charts for the loaded dataset. When only the bundled sample is available, the app must label it as a limited preview.

### 4. Model Performance

Displays precision, recall, F1, Average Precision, confusion matrix, and threshold information. These evaluation metrics should not be recalculated from the 100-row preview.

### 5. Model Details

Displays model identity, feature count, threshold, artifact status, and limitations.

### Why Streamlit?

Streamlit turns Python code into an interactive web application without requiring a separate frontend and backend. It is useful for an educational demo, although a production system may need a different serving architecture.

---

## 14. Step 12 — Handle the Missing Full Dataset

The full `creditcard.csv` is approximately 144 MB and is intentionally excluded from the ordinary GitHub repository.

Initially, the deployed app only looked for the complete CSV. Because the file was not present in the cloud deployment, sample selection and Dataset Explorer were unavailable.

### The fix

The project includes:

`data/sample_transactions.csv`

The bundled file contains 100 real rows selected from the local dataset. Its recorded composition is 80 legitimate and 20 fraud-labelled rows.

The intended dataset-resolution priority is:

1. Use the complete `creditcard.csv` if available.
2. Otherwise, use `data/sample_transactions.csv`.
3. If neither file exists, show a clear message rather than crashing.

### Why not upload the full CSV to GitHub?

The approximately 144 MB file exceeds GitHub's normal 100 MiB individual-file limit. Large files also increase repository and deployment overhead.

### What is the preview for?

It enables sample selection and preview charts. It must not be used to claim full-dataset statistics or to calculate the reported held-out test metrics.

The preview intentionally has a higher fraud proportion than the original dataset, so the UI must explain that its distribution is not representative of the original dataset's fraud rate.

### What happens during a sample prediction?

1. Read the 30 feature values.
2. Exclude `Class` from the model input.
3. Ask the saved model for a fraud score.
4. Apply the threshold.
5. Display the prediction and, separately, the historical dataset label.

A correct model can disagree with a sample's label. That is a model error for that example, not automatically an app bug.

---

## 15. Step 13 — Test and Deploy

### Automated tests

The previous development report stated that the suite had **63 tests passed** using:

```bash
python -B -m unittest discover -s tests -v
```

The full audit should rerun this command and inspect actual coverage. Passing tests do not prove every input and cloud scenario works.

### GitHub hygiene

The `.gitignore` is intended to keep the full dataset and local development files out of GitHub, while the small preview remains tracked.

Before publishing changes, review:

```bash
git status
git diff
git diff --cached
```

Ensure the full dataset, credentials, environment files, caches, and unrelated files are not staged.

### Deployment

- GitHub repository: https://github.com/AfrozBadra/FraudShield-AI
- Streamlit app: https://payshield-ml.streamlit.app/
- App entry point: `10_app.py`

### Why deployment matters

Cloud environments may not contain files, cached data, or libraries available on a developer's laptop. Correct relative paths, dependency compatibility, and missing-file handling are essential.

The sample fallback addressed the missing-CSV problem. The current audit still needs to verify all live pages and the inference pipeline.

---

## 16. How the Website Makes a Prediction

```text
User selects a sample or enters custom features
                       |
                       v
Validate input and required columns
                       |
                       v
Select the 30 features in training order
                       |
                       v
Pass features to the saved Random Forest
                       |
                       v
Get probability for fraud class 1
                       |
                       v
Apply threshold 0.20
                       |
             +---------+---------+
             |                   |
        Probability < 0.20   Probability >= 0.20
             |                   |
             v                   v
       Predict legitimate    Predict fraud
             |                   |
             +---------+---------+
                       |
                       v
Display prediction and probability
```

This is the intended logic; the audit must verify the implementation.

### Why validate input?

Custom values can be missing, non-numeric, or malformed. Validation prevents avoidable crashes and misleading predictions.

### Why use a shared feature schema?

Training and inference must agree on feature names, ordering, data types, and preprocessing.

### Why keep the label separate?

A new transaction's true label is generally unknown when the model makes its prediction. The model must make the prediction without seeing the answer.

---

## 17. How to Interpret the Metrics

| Metric | Question answered | Limitation |
|---|---|---|
| Precision | How reliable are fraud alerts? | Can fall as false alarms increase |
| Recall | How much fraud is detected? | Can rise while false alarms increase |
| F1 | How do precision and recall balance? | Does not directly include financial costs |
| Average Precision | How well are fraud examples ranked across thresholds? | Does not choose the operating threshold |
| Confusion matrix | What types of errors occur? | Depends on threshold and evaluation sample |
| Accuracy | What fraction of all predictions are correct? | Often misleading for rare fraud |

The best threshold depends on the cost of missed fraud, the cost of false alarms, review capacity, and validation evidence. A threshold of 0.20 is the project's current setting, not a universal rule.

---

## 18. Limitations and Responsible Claims

1. **Public historical data:** Performance does not guarantee performance on current transactions.
2. **Anonymized features:** The original meanings of `V1`–`V28` are not provided.
3. **Class imbalance:** Evaluation must account for rare positive examples.
4. **Threshold trade-off:** Changing the threshold changes false alarms and missed fraud.
5. **Probability calibration:** A predicted score is not automatically a calibrated real-world probability.
6. **Sample preview:** The 100-row preview is not the full dataset and has a deliberately different fraud proportion.
7. **No live banking integration:** The app is not connected to a bank or payment processor.
8. **No demonstrated production monitoring:** Real use would need drift monitoring, reliability checks, security controls, and operational review.
9. **Evaluation integrity:** The test-set procedure and metric reproducibility still need audit confirmation.
10. **Artifact trust:** Joblib/pickle-based artifacts should only be loaded from trusted sources.

Do not claim the app prevents all fraud, guarantees financial security, or is production-ready.

---

## 19. Interview-Ready Explanation

> I built FraudShield AI, an end-to-end credit-card fraud detection project using supervised machine learning. The dataset is highly imbalanced, so I focused on precision, recall, F1 score, Average Precision, and the confusion matrix rather than relying only on accuracy.
>
> I inspected and prepared the data, compared classification models, and selected a Random Forest model for the deployed application. I used a threshold of 0.20 to convert the model's fraud score into a class prediction, balancing fraud detection against false alarms.
>
> I saved the model with joblib and built a Streamlit dashboard with transaction risk analysis, custom feature input, dataset exploration, model performance, and model details. I deployed the app using GitHub and Streamlit Community Cloud.
>
> The cloud deployment initially could not find the complete CSV, so I added a small bundled preview and fallback logic to keep sample selection working without committing the large dataset.
>
> This project taught me that ML engineering involves more than training a classifier: it requires sound evaluation, consistent inference features, robust data handling, testing, deployment, and honest communication of limitations. It is an educational prototype, not a production banking security system.

**Before presenting the reported metrics as final, confirm them against the completed audit.**

---

## 20. Interview Questions and Answers

### Q1. Why is accuracy misleading for fraud detection?

Because almost all transactions are legitimate. A model can predict every transaction as legitimate and still achieve high accuracy while detecting zero fraud.

### Q2. Why use precision and recall?

Precision measures how many fraud alerts are correct; recall measures how much actual fraud is detected. Both matter because false alarms and missed fraud have different consequences.

### Q3. What is a false positive?

A legitimate transaction incorrectly predicted as fraud. It may trigger unnecessary alerts or manual review.

### Q4. What is a false negative?

A fraudulent transaction incorrectly predicted as legitimate. It is a missed fraud case.

### Q5. Why use a threshold of 0.20?

It is the current app setting. Lower thresholds usually catch more fraud but generate more false alarms. The threshold should be justified using validation data and intended operating costs.

### Q6. Why Random Forest?

It can learn nonlinear patterns and interactions through an ensemble of decision trees. The choice should be supported by actual model-comparison results, not by assuming it is always best.

### Q7. Why save the model?

So the deployed app can reuse the fitted model without retraining on every request.

### Q8. What is target leakage?

When information unavailable at prediction time influences training or evaluation. Feeding `Class` into the input features is a direct example.

### Q9. Why do feature order and schema matter?

The model expects a consistent input format. Incorrect ordering or missing columns can cause errors or invalid predictions.

### Q10. Why not upload the full dataset to GitHub?

The approximately 144 MB file exceeds GitHub's ordinary 100 MiB individual-file limit. The project uses a small bundled preview instead.

### Q11. Does the preview represent the original fraud rate?

No. It contains 80 legitimate and 20 fraud-labelled examples by design. It demonstrates the UI, not the original class distribution.

### Q12. Does the dataset label determine the model prediction?

No. The model should receive only the 30 features. `Class` is the historical answer and is shown separately for comparison.

### Q13. Is the app production-ready?

No. It is an educational prototype. Production use would require more validation, calibration, monitoring, security, integration, and governance.

### Q14. What would you improve next?

After the audit, strengthen evaluation methodology, test calibration, improve input and integration tests, add observability, and plan monitoring for changes in incoming data.

---

## 21. ML Engineer Lessons

### Data engineering
- Inspect schemas, data types, duplicates, missing values, and target distributions.
- Keep the feature schema consistent.
- Document data sources and transformations.

### Modelling
- Establish a baseline.
- Compare algorithms with suitable metrics.
- Tune using validation data.
- Avoid repeated test-set tuning.

### Evaluation
- Understand each metric.
- Inspect the confusion matrix.
- Reproduce results from code and artifacts.
- Communicate limitations honestly.

### Software engineering
- Save and reload the fitted model.
- Separate data loading, validation, inference, and display logic where practical.
- Handle missing files and invalid inputs.
- Use tests to catch regressions.

### Deployment and MLOps
- Keep large datasets and secrets out of source control.
- Document compatible dependencies.
- Test the cloud environment, not only the local machine.
- Track model versions and thresholds.
- For production systems, plan monitoring, drift checks, rollback, and incident handling.

### Connection to AI Engineering

This project teaches the lifecycle of a model-backed application. The same discipline is useful for LLM applications, RAG systems, and agents: prepare data, evaluate behavior, serve the model reliably, test failure cases, monitor the system, and communicate limitations.

---

## 22. Final Checklist

### Data and modelling
- [x] Dataset inspection performed during development
- [x] Feature and target definitions established
- [x] Class imbalance recognized
- [x] Multiple classification models explored
- [x] Random Forest artifact saved
- [x] Threshold documented as 0.20
- [ ] Audit verifies training and evaluation methodology
- [ ] Audit reproduces metrics using the correct artifact and test data
- [ ] Audit verifies fraud probability column and feature ordering

### Application
- [x] Streamlit dashboard deployed
- [x] Sample-preview fallback added
- [x] Full CSV kept out of ordinary GitHub repository
- [ ] Audit confirms all pages and interactions work
- [ ] Audit verifies custom-input validation and inference consistency

### Engineering
- [x] Source and README published on GitHub
- [x] Automated test suite exists
- [ ] Rerun tests and review actual coverage
- [ ] Complete security and dependency review
- [ ] Document and fix remaining issues

### Final takeaway

A successful ML project is not just a trained model. It is a repeatable workflow: understand the data, avoid leakage, train and compare models, evaluate the right metrics, choose a threshold, save the model, build a reliable inference pipeline, test it, deploy it, and explain its limitations honestly.

The goal is not to claim FraudShield AI is perfect. The goal is to understand why it works, know where it may fail, and defend engineering decisions with evidence.
