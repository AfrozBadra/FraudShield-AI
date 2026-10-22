# Step 5 — Model comparison
# We'll compare three models:
# - Logistic Regression — our existing benchmark.
# - Random Forest — learns more complex, nonlinear patterns.
# - HistGradientBoosting — builds trees sequentially to improve on previous errors.
# We'll use the same training and validation split for a fair comparison. The test set remains untouched.




# FraudShield AI — Step 5: Model Comparison

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier #HistGradientBoostingClassifier is a gradient boosting algorithm that builds an ensemble of decision trees sequentially, where each new tree corrects the errors made by the previous trees. It is designed to handle large datasets efficiently and can capture complex patterns in the data.
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)

# 1. Load data and remove exact duplicates
# WHY: Every model must learn from the same cleaned dataset.
data_path = Path(__file__).resolve().parent / "creditcard.csv"
df = pd.read_csv(data_path).drop_duplicates().copy()

X = df.drop(columns=["Class"])
y = df["Class"]

# 2. Recreate the same 70/15/15 split
# WHY: Same random seed and row order give us consistent splits.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# 3. Define models
# WHY: Different algorithms may detect different fraud patterns.
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        class_weight="balanced",
        random_state=42
    ),
}

results = []

# 4. Train each model and evaluate on validation data
# WHY: Validation results help us select a model without using the test set.
for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    # Fraud probability = probability of Class 1
    val_proba = model.predict_proba(X_val)[:, 1]

    # Keep the same 0.50 threshold for this first comparison
    val_pred = (val_proba >= 0.50).astype(int)

    precision = precision_score(y_val, val_pred, zero_division=0)
    recall = recall_score(y_val, val_pred, zero_division=0)
    f1 = f1_score(y_val, val_pred, zero_division=0)
    avg_precision = average_precision_score(y_val, val_proba)

    results.append({
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Average Precision": avg_precision,
    })

    print("Training complete!")

# 5. Display a comparison table
comparison = pd.DataFrame(results)
comparison = comparison.sort_values(
    by="Average Precision", ascending=False
)

print("\nFRAUDSHIELD AI — MODEL COMPARISON")
print("---------------------------------")
print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
