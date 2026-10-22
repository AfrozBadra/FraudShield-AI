
# FraudShield AI — Step 2: Data Preparation

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load the dataset
data_path = Path(__file__).resolve().parent / "creditcard.csv"
df = pd.read_csv(data_path)

print("Original dataset shape:", df.shape)

# 2. Remove exact duplicate rows
# WHY: Avoid repeatedly counting identical records in our modeling data.
df_model = df.drop_duplicates().copy()

print("After removing duplicates:", df_model.shape)
print("Remaining duplicate rows:", df_model.duplicated().sum())

# 3. Separate features and target
# X = information used to predict fraud
# y = the actual answer we want to predict
X = df_model.drop(columns=["Class"])
y = df_model["Class"]

print("\nFeature matrix shape:", X.shape)
print("Target shape:", y.shape)

# 4. Split into 70% training and 30% temporary data
# Stratification preserves the approximate fraud percentage.
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

# 5. Split the remaining 30% equally:
# 15% validation and 15% testing overall.
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

# 6. Verify the split sizes and fraud counts
print("\nTraining set:", X_train.shape,
      "| Fraud cases:", int(y_train.sum()))

print("Validation set:", X_val.shape,
      "| Fraud cases:", int(y_val.sum()))

print("Testing set:", X_test.shape,
      "| Fraud cases:", int(y_test.sum()))

print("\nFraud percentage:")
print(f"Training:   {y_train.mean() * 100:.4f}%")
print(f"Validation: {y_val.mean() * 100:.4f}%")
print(f"Testing:    {y_test.mean() * 100:.4f}%")

print("\nTotal rows across splits:",
      len(X_train) + len(X_val) + len(X_test))
