
# FraudShield AI — Logistic Regression
# Step 4: Train our first real fraud detection model

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)

# 1. Load and prepare the dataset
# WHY: We remove exact duplicate transactions so repeated rows
# don't unnecessarily influence model training.
data_path = Path(__file__).resolve().parent / "creditcard.csv"
df = pd.read_csv(data_path)
df = df.drop_duplicates().copy()

X = df.drop(columns=["Class"])
y = df["Class"]

# 2. Create the same 70% / 15% / 15% split as before
# WHY: Keeping the same random seed and split process makes
# comparisons between our models fair.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Training fraud cases:", int(y_train.sum()))
print("Validation fraud cases:", int(y_val.sum()))

# 3. Build our ML pipeline
# WHY: Scaling helps Logistic Regression handle features
# with different numerical ranges. The pipeline learns the
# scaling values from training data only.
# class_weight='balanced' gives the rare fraud class more weight.
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

# 4. Train the model
# WHY: The model learns patterns that help distinguish
# fraudulent transactions from legitimate ones.
print("\nTraining Logistic Regression...")
model.fit(X_train, y_train)
print("Training complete!")

# 5. Predict fraud probabilities on validation data
# WHY: Probabilities let us evaluate ranking quality and later
# experiment with different fraud-alert thresholds.
y_val_proba = model.predict_proba(X_val)[:, 1]

# Start with the standard 0.50 threshold
y_val_pred = (y_val_proba >= 0.50).astype(int)

# 6. Evaluate using metrics suited to rare fraud cases
precision = precision_score(y_val, y_val_pred, zero_division=0)
recall = recall_score(y_val, y_val_pred, zero_division=0)
f1 = f1_score(y_val, y_val_pred, zero_division=0)
avg_precision = average_precision_score(y_val, y_val_proba)

print("\nFRAUDSHIELD AI — LOGISTIC REGRESSION")
print("------------------------------------")
print(f"Validation precision:      {precision:.4f}")
print(f"Validation recall:         {recall:.4f}")
print(f"Validation F1-score:       {f1:.4f}")
print(f"Validation Average Precision: {avg_precision:.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(y_val, y_val_pred))


#training = learning, validation = practice exam, test = final exam