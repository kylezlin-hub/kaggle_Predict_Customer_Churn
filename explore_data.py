import pandas as pd

try:
    train = pd.read_csv('c:/kaggle_Predict_Customer_Churn/data/train.csv')
    test = pd.read_csv('c:/kaggle_Predict_Customer_Churn/data/test.csv')
except Exception as e:
    print(f"Error reading CSVs: {e}")
    exit(1)

print("=== TRAIN HEAD ===")
print(train.head(2).to_string())

print("\n=== COLUMNS DIFFERENCE (Target) ===")
target_cols = set(train.columns) - set(test.columns)
print(f"Target column(s): {target_cols}")

print("\n=== TRAIN MISSING VALUES ===")
missing = train.isnull().sum()
print(missing[missing > 0])

print("\n=== TEST MISSING VALUES ===")
missing_test = test.isnull().sum()
print(missing_test[missing_test > 0])

print("\n=== TARGET DISTRIBUTION ===")
for col in target_cols:
    print(train[col].value_counts(dropna=False))

print("\n=== NUMERICAL DESCRIBE ===")
print(train.describe().T.to_string())

print("\n=== CATEGORICAL DESCRIBE ===")
categorical = train.select_dtypes(include=['object'])
if not categorical.empty:
    print(categorical.describe().T.to_string())
else:
    print("No categorical columns detected by default types.")
