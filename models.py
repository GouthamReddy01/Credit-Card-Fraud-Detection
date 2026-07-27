import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load dataset
data = pd.read_csv("fraud_dataset_25k_mixed.csv")


# Separate features and target
X = data.drop("fraud", axis=1)
y = data["fraud"]


# Define columns

numerical_cols = [
    "transaction_amount",
    "transaction_hour",
    "location_distance",
    "daily_transaction_count",
    "avg_transaction_7d",
    "account_age_months",
    "declined_transactions"
]


categorical_cols = [
    "transaction_type",
    "merchant_category",
    "card_present",
    "foreign_transaction",
    "device_change",
    "ip_change",
    "weekend_transaction"
]


# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Scale numerical features

scaler = StandardScaler()

X_train_num = scaler.fit_transform(X_train[numerical_cols])
X_test_num = scaler.transform(X_test[numerical_cols])


# Encode categorical features

encoder = OneHotEncoder(handle_unknown="ignore")

X_train_cat = encoder.fit_transform(X_train[categorical_cols]).toarray()
X_test_cat = encoder.transform(X_test[categorical_cols]).toarray()


# Combine numerical + categorical

import numpy as np

X_train_final = np.concatenate((X_train_num, X_train_cat), axis=1)
X_test_final = np.concatenate((X_test_num, X_test_cat), axis=1)
print(X_train_final.shape, X_test_final.shape)


# Train model

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train_final, y_train)


# Predict

y_pred = model.predict(X_test_final)


# Accuracy

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)


# Save model and preprocessors

joblib.dump(model, "fraud_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "encoder.pkl")

print("Model, scaler, and encoder saved.")