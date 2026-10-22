# Step 9 — Verify the saved model
# Before building a dashboard, we should prove that the saved artifact loads correctly and generates predictions. This is an important deployment check: it tests the actual saved file, not just the model still in memory.




# FraudShield AI — Step 9: Test the Saved Model

from pathlib import Path
import joblib
import pandas as pd

# 1. Locate and load the saved artifact
# WHY: Verify that the saved model can be reused without retraining.
project_dir = Path(__file__).resolve().parent
model_path = project_dir / "models" / "fraudshield_random_forest.joblib"

artifact = joblib.load(model_path)

model = artifact["model"]
threshold = artifact["threshold"]
feature_names = artifact["feature_names"]

print("Saved model loaded successfully!")
print("Threshold:", threshold)
print("Expected features:", len(feature_names))

# 2. Load a few real transactions from the dataset
# WHY: Check predictions using real data with the correct columns.
df = pd.read_csv(project_dir / "creditcard.csv")
df = df.drop_duplicates().copy()

X = df[feature_names]
y = df["Class"]

# Select a few examples, including fraud if available.
fraud_indices = y[y == 1].index[:3]
legitimate_indices = y[y == 0].index[:3]
sample_indices = list(legitimate_indices) + list(fraud_indices)

X_sample = X.loc[sample_indices]
y_sample = y.loc[sample_indices]

# 3. Generate fraud probabilities
# WHY: The model returns probabilities; our saved threshold
# converts them into fraud or legitimate predictions.
probabilities = model.predict_proba(X_sample)[:, 1]
predictions = (probabilities >= threshold).astype(int)

# 4. Display the results
results = pd.DataFrame({
    "Actual Class": y_sample.to_numpy(),
    "Fraud Probability": probabilities,
    "Predicted Class": predictions,
})

results["Prediction"] = results["Predicted Class"].map({
    0: "Legitimate",
    1: "Fraud Alert",
})

print("\nSAVED MODEL PREDICTIONS")
print("-----------------------")
print(results.to_string(
    index=False,
    formatters={"Fraud Probability": "{:.4%}".format}
))
