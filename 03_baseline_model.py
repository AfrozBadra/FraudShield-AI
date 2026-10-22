from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier #DummyClassifier is a simple baseline classifier that makes predictions using simple rules, such as always predicting the most frequent class. It is often used as a benchmark to compare the performance of more complex models.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# 1. Load the dataset
df = pd.read_csv("/home/badra-pc/Desktop/python-AI-learning/FraudShield AI/creditcard.csv")

# 2. Remove exact duplicate rows
df_model = df.drop_duplicates().copy()

# 3. Separate input features and target
X = df_model.drop(columns=["Class"])
y = df_model["Class"]

# 4. Create a 70/15/15 stratified split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42,
)

# 5. Train the majority-class baseline
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

# 6. Evaluate the baseline
y_pred = baseline.predict(X_test)

print("FRAUDSHIELD AI — BASELINE RESULTS")
print("---------------------------------")
print(f"Test accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision:      {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall:         {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-score:       {f1_score(y_test, y_pred, zero_division=0):.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))