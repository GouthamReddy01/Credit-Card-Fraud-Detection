from django.shortcuts import render
import joblib
import numpy as np
import pandas as pd


# Load trained components
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")


def home(request):

    result = ""

    if request.method == "POST":

        # Numerical inputs
        transaction_amount = float(request.POST["transaction_amount"])
        transaction_hour = float(request.POST["transaction_hour"])
        location_distance = float(request.POST["location_distance"])
        daily_transaction_count = float(request.POST["daily_transaction_count"])
        avg_transaction_7d = float(request.POST["avg_transaction_7d"])
        account_age_months = float(request.POST["account_age_months"])
        declined_transactions = float(request.POST["declined_transactions"])


        # Categorical inputs
        transaction_type = request.POST["transaction_type"]
        merchant_category = request.POST["merchant_category"]
        card_present = request.POST["card_present"]
        foreign_transaction = request.POST["foreign_transaction"]
        device_change = request.POST["device_change"]
        ip_change = request.POST["ip_change"]
        weekend_transaction = request.POST["weekend_transaction"]


        # Numerical preprocessing
        numerical_data = [[
            transaction_amount,
            transaction_hour,
            location_distance,
            daily_transaction_count,
            avg_transaction_7d,
            account_age_months,
            declined_transactions
        ]]

        numerical_scaled = scaler.transform(numerical_data)


        # Categorical preprocessing
        categorical_data = [[
            transaction_type,
            merchant_category,
            card_present,
            foreign_transaction,
            device_change,
            ip_change,
            weekend_transaction
        ]]

        categorical_encoded = encoder.transform(categorical_data).toarray()


        # Combine features
        final_input = np.concatenate((numerical_scaled, categorical_encoded), axis=1)


        # Prediction
        prediction = model.predict(final_input)[0]


        if prediction == 1:
            result = "⚠️ Fraud Transaction Detected"
        else:
            result = "✅ Genuine Transaction"


    return render(request, "index.html", {"result": result})