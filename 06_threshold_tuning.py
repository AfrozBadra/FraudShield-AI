# Step 6 — Threshold Tuning
# Our Step 5 comparison showed that Random Forest is currently our strongest model at threshold 0.50. Now we're going to investigate whether we can catch more fraudulent transactions without creating too many false alarms.



# FraudShield AI — Step 6: Random Forest Threshold Tuning

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# STEP 1: Load and clean the dataset
# WHY: Use the same data preparation as Step 5 for a fair experiment.
data_path = Path(__file__).resolve().parent / "creditcard.csv"
df = pd.read_csv(data_path).drop_duplicates().copy()

X = df.drop(columns=["Class"])
y = df["Class"]

# STEP 2: Recreate our same 70/15/15 split
# WHY: Train on the same rows and tune using the same validation rows.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# STEP 3: Train our current best model
# WHY: Random Forest had the strongest results in Step 5.
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest...")
model.fit(X_train, y_train)
print("Training complete!")

# STEP 4: Get fraud probabilities ONCE
# WHY: The probabilities stay fixed; only the threshold changes.
y_val_proba = model.predict_proba(X_val)[:, 1]

# STEP 5: Test different thresholds
# WHY: We want to understand the trade-off between catching fraud
# and raising false alarms.
thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

results = []

for threshold in thresholds:
    y_val_pred = (y_val_proba >= threshold).astype(int)

    precision = precision_score(
        y_val, y_val_pred, zero_division=0
    )
    recall = recall_score(
        y_val, y_val_pred, zero_division=0
    )
    f1 = f1_score(
        y_val, y_val_pred, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_val, y_val_pred, labels=[0, 1]
    ).ravel()

    results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "False Alarms (FP)": fp,
        "Missed Fraud (FN)": fn,
        "Fraud Caught (TP)": tp,
    })

# STEP 6: Display the results
comparison = pd.DataFrame(results)

print("\nFRAUDSHIELD AI — THRESHOLD COMPARISON")
print("-------------------------------------")
print(comparison.to_string(
    index=False,
    formatters={
        "Threshold": "{:.2f}".format,
        "Precision": "{:.4f}".format,
        "Recall": "{:.4f}".format,
        "F1-score": "{:.4f}".format,
    }
))

# Show the threshold with the highest F1-score
# WHY: F1 is a useful starting point for balancing precision and recall.
best_row = comparison.loc[comparison["F1-score"].idxmax()]

print("\nBEST THRESHOLD BY F1-SCORE")
print("--------------------------")
print(f"Threshold:       {best_row['Threshold']:.2f}")
print(f"Precision:       {best_row['Precision']:.4f}")
print(f"Recall:          {best_row['Recall']:.4f}")
print(f"F1-score:        {best_row['F1-score']:.4f}")
print(f"False alarms:    {int(best_row['False Alarms (FP)'])}")
print(f"Missed fraud:    {int(best_row['Missed Fraud (FN)'])}")
print(f"Fraud caught:    {int(best_row['Fraud Caught (TP)'])}")
