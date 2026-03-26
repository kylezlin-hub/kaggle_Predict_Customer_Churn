import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score

print("Loading data...")
train = pd.read_csv("c:/kaggle_Predict_Customer_Churn/data/train.csv")
test = pd.read_csv("c:/kaggle_Predict_Customer_Churn/data/test.csv")


def feature_engineering(df):
    df = df.copy()
    eps = 1e-3

    # Ratio & Differences
    df["charges_per_tenure"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["monthly_diff"] = df["MonthlyCharges"] - df["charges_per_tenure"]

    # Binning Tenure
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 36, 48, 60, 72, 100],
        labels=["0-12", "13-24", "25-36", "37-48", "49-60", "61-72", "73+"],
    )
    df["tenure_group"] = df["tenure_group"].astype(str)

    # Additional services count
    services = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    df["num_services"] = df[services].apply(lambda x: (x == "Yes").sum(), axis=1)

    # Feature Combinations
    df["contract_internet"] = df["Contract"] + "_" + df["InternetService"]
    df["payment_paperless"] = df["PaymentMethod"] + "_" + df["PaperlessBilling"]

    df["avg_spend"] = df["TotalCharges"] / (df["tenure"] + eps)
    df["tenure_x_monthly"] = df["tenure"] * df["MonthlyCharges"]
    df["charges_per_tenure"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["charges_diff"] = df["TotalCharges"] - df["MonthlyCharges"] * df["tenure"]
    df["monthly_to_total_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

    # Cost relationships
    df["per_cost"] = df["MonthlyCharges"] / (df["num_services"] + eps)
    df["tenure_per_service"] = df["tenure"] / (df["num_services"] + 1)

    # Contract features
    df["monthly_contract"] = (df["Contract"] == "Month-to-month").astype(int)
    df["long_contract"] = df["Contract"].isin(["One year", "Two year"]).astype(int)

    # Payment / risk-style flags
    df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["is_auto_payment"] = (
        df["PaymentMethod"]
        .isin(["Bank transfer (automatic)", "Credit card (automatic)"])
        .astype(int)
    )
    df["paperless_and_echeck"] = (
        (df["PaperlessBilling"] == "Yes") & (df["PaymentMethod"] == "Electronic check")
    ).astype(int)

    # Internet features
    df["no_internet"] = (df["InternetService"] == "No").astype(int)

    # Customer lifecycle
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
    df["is_loyal_customer"] = (df["tenure"] > 24).astype(int)

    # Senior + dependents / partner interactions
    df["senior_no_partner"] = (
        (df["SeniorCitizen"] == 1) & (df["Partner"] == "No")
    ).astype(int)
    df["senior_no_dependents"] = (
        (df["SeniorCitizen"] == 1) & (df["Dependents"] == "No")
    ).astype(int)

    # Streaming bundle
    df["streaming_bundle"] = (df["StreamingTV"] == "Yes").astype(int) + (
        df["StreamingMovies"] == "Yes"
    ).astype(int)

    return df


print("Applying Feature Engineering...")
train = feature_engineering(train)
test = feature_engineering(test)

# Prepare features
X = train.drop(["id", "Churn"], axis=1)
y = train["Churn"].map({"Yes": 1, "No": 0})
X_test = test.drop(["id"], axis=1)

# Column groups
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print(f"Categorical features: {len(cat_cols)}")
print(f"Numerical features: {len(num_cols)}")

cat_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

# Transforming all columns to a single representation
# The categorical columns are placed first
preprocessor = ColumnTransformer(
    transformers=[("cat", cat_preprocessor, cat_cols)], remainder="passthrough"
)

# HistGB expects categorical columns by indices, which are the first len(cat_cols) columns
categorical_features_indices = list(range(len(cat_cols)))

model = HistGradientBoostingClassifier(
    loss="log_loss",
    categorical_features=categorical_features_indices,
    learning_rate=0.05,
    max_iter=300,
    random_state=42,
)

pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

print("Running 5-Fold Stratified CV...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=1)

print(f"CV ROC-AUC Scores: {scores}")
print(f"Mean ROC-AUC: {np.mean(scores):.5f} (+/- {np.std(scores):.5f})")

print("Training on full dataset...")
pipeline.fit(X, y)

print("Predicting on test set...")
# predict_proba outputs [class_0_prob, class_1_prob]
# We want the probability of class 1 ('Yes')
test_probs = pipeline.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({"id": test["id"], "Churn": test_probs})

sub_path = "c:/kaggle_Predict_Customer_Churn/submission2.csv"
submission.to_csv(sub_path, index=False)
print(f"Saved submission to {sub_path}")
print("Done!")
