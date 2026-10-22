
# FraudShield AI — Step 1: Data Inspection

from pathlib import Path
import pandas as pd

# Load the dataset located beside this script
data_path = Path(__file__).resolve().parent / "creditcard.csv"
df = pd.read_csv(data_path)

# 1. Dataset size
print("Dataset shape:", df.shape)

# 2. Column names and data types
print("\nDataset information:")
df.info()

# 3. Missing values
print("\nTotal missing values:", df.isnull().sum().sum())

# 4. Duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())

# 5. Class distribution
print("\nTransaction classes:")
print(df["Class"].value_counts().sort_index())

print("\nClass percentages:")
print((df["Class"].value_counts(normalize=True) * 100).round(4))

# 6. Understand our target labels
print("\nTarget meaning:")
print("Class 0 = Legitimate transaction")
print("Class 1 = Fraudulent transaction")
