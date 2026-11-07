
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score
)

# 1. Load the dataset
df = pd.read_csv("data/creditcard.csv")

# 2. Separate input features and target
X = df.drop(columns=["Class"])
y = df["Class"]

# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

import os

os.makedirs("data", exist_ok=True)

test_data = X_test.copy()
test_data["Class"] = y_test

test_data.to_csv("data/test_transactions.csv", index=False)

print("Test dataset saved successfully!")

# 4. Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

# 5. Train the model
print("Training model...")
model.fit(X_train, y_train)

# 6. Make predictions on unseen test data
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 7. Evaluate the model
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Legitimate", "Fraud"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(
    "Average Precision:",
    round(average_precision_score(y_test, y_prob), 4)
)

# 8. Save the trained model
joblib.dump(model, "models/fraud_model.pkl")

print("\nModel saved successfully!")
