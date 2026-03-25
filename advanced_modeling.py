import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")


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
train_fe = feature_engineering(train)
test_fe = feature_engineering(test)

X = train_fe.drop(["id", "Churn"], axis=1)
y = train_fe["Churn"].map({"Yes": 1, "No": 0})
X_test = test_fe.drop(["id"], axis=1)

cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

for c in cat_features:
    X[c] = X[c].astype(str).fillna("missing")
    X_test[c] = X_test[c].astype(str).fillna("missing")

print(f"Total Features: {X.shape[1]}")
print(f"Categorical features: {len(cat_features)}")

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

cv_scores = []

print("Training CatBoost models...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        task_type="CPU",
        verbose=100,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        use_best_model=True,
    )

    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    fold_auc = roc_auc_score(y_val, val_preds)
    cv_scores.append(fold_auc)
    print(f"Fold {fold+1} AUC: {fold_auc:.5f}")

    test_preds += model.predict_proba(X_test)[:, 1] / n_splits

mean_auc = np.mean(cv_scores)
print(f"\nCV AUC Scores: {cv_scores}")
print(f"Mean CV AUC: {mean_auc:.5f} (+/- {np.std(cv_scores):.5f})")

submission = pd.DataFrame({"id": test["id"], "Churn": test_preds})
sub_path = "c:/kaggle_Predict_Customer_Churn/submission_advanced2.csv"
submission.to_csv(sub_path, index=False)
print(f"Saved advanced submission to {sub_path}")
