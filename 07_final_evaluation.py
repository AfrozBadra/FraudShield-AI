# Step 7 — Final evaluation on the test set
# Now, bro, we need to check whether our selected approach performs well on data that wasn't used to choose the threshold.



# FraudShield AI — Step 7: Final Test Evaluation

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# 1. Load and prepare data
data_path = Path(__file__).resolve().parent / "creditcard.csv"
df = pd.read_csv(data_path).drop_duplicates().copy()

X = df.drop(columns=["Class"])
y = df["Class"]

# 2. Recreate the same data split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# 3. Train the selected model on training data only
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("Training final Random Forest model...")
model.fit(X_train, y_train)
print("Training complete!")

# 4. Apply the threshold selected using validation data
# WHY: We chose 0.20 based on validation F1-score.
# We must not select a new threshold using test results.
FINAL_THRESHOLD = 0.20

y_test_proba = model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= FINAL_THRESHOLD).astype(int)

# 5. Calculate final test metrics
precision = precision_score(y_test, y_test_pred, zero_division=0)
recall = recall_score(y_test, y_test_pred, zero_division=0)
f1 = f1_score(y_test, y_test_pred, zero_division=0)
avg_precision = average_precision_score(y_test, y_test_proba)

tn, fp, fn, tp = confusion_matrix(
    y_test, y_test_pred, labels=[0, 1]
).ravel()

print("\nFRAUDSHIELD AI — FINAL TEST RESULTS")
print("-----------------------------------")
print(f"Threshold:          {FINAL_THRESHOLD:.2f}")
print(f"Precision:          {precision:.4f}")
print(f"Recall:             {recall:.4f}")
print(f"F1-score:           {f1:.4f}")
print(f"Average Precision:  {avg_precision:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nConfusion Matrix Breakdown:")
print(f"True negatives  (legitimate correctly identified): {tn}")
print(f"False positives (legitimate flagged as fraud):     {fp}")
print(f"False negatives (fraud missed):                    {fn}")
print(f"True positives  (fraud correctly caught):          {tp}")

print("\nDetailed Classification Report:")
print(classification_report(
    y_test, y_test_pred,
    target_names=["Legitimate", "Fraud"],
    zero_division=0
))
