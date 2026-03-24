import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score

print("Loading data...")
train = pd.read_csv('c:/kaggle_Predict_Customer_Churn/data/train.csv')
test = pd.read_csv('c:/kaggle_Predict_Customer_Churn/data/test.csv')

# Prepare features
X = train.drop(['id', 'Churn'], axis=1)
y = train['Churn'].map({'Yes': 1, 'No': 0})
X_test = test.drop(['id'], axis=1)

# Column groups
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

print(f"Categorical features: {len(cat_cols)}")
print(f"Numerical features: {len(num_cols)}")

cat_preprocessor = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Transforming all columns to a single representation
# The categorical columns are placed first
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', cat_preprocessor, cat_cols)
    ],
    remainder='passthrough'
)

# HistGB expects categorical columns by indices, which are the first len(cat_cols) columns
categorical_features_indices = list(range(len(cat_cols)))

model = HistGradientBoostingClassifier(
    loss='log_loss',
    categorical_features=categorical_features_indices,
    learning_rate=0.05,
    max_iter=300,
    random_state=42
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

print("Running 5-Fold Stratified CV...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)

print(f"CV ROC-AUC Scores: {scores}")
print(f"Mean ROC-AUC: {np.mean(scores):.5f} (+/- {np.std(scores):.5f})")

print("Training on full dataset...")
pipeline.fit(X, y)

print("Predicting on test set...")
# predict_proba outputs [class_0_prob, class_1_prob]
# We want the probability of class 1 ('Yes')
test_probs = pipeline.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    'id': test['id'],
    'Churn': test_probs
})

sub_path = 'c:/kaggle_Predict_Customer_Churn/submission.csv'
submission.to_csv(sub_path, index=False)
print(f"Saved submission to {sub_path}")
print("Done!")
