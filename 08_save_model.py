
# FraudShield AI — Step 8: Save Model for Deployment

from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Load and prepare the data
# WHY: Reproduce the same training split and model configuration
# that generated our final test results.
project_dir = Path(__file__).resolve().parent
df = pd.read_csv(project_dir / "creditcard.csv")
df = df.drop_duplicates().copy()

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# 2. Train the final evaluated model configuration
# WHY: The saved model should match the model configuration
# used in our final test evaluation.
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("Training model for saving...")
model.fit(X_train, y_train)

# 3. Save model + threshold + expected input features together
# WHY: Future predictions must use the same threshold and
# exactly the same feature names and order.
artifact = {
    "model": model,
    "threshold": 0.20,
    "feature_names": list(X.columns),
}

models_dir = project_dir / "models"
models_dir.mkdir(exist_ok=True)

model_path = models_dir / "fraudshield_random_forest.joblib"
joblib.dump(artifact, model_path)

print("\nModel saved successfully!")
print(f"Location: {model_path}")
print(f"Threshold: {artifact['threshold']}")
print(f"Number of input features: {len(artifact['feature_names'])}")
