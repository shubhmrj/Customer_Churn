---
title: Customer Churn Prediction
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.32.2
app_file: app.py
pinned: false
---

# Customer Churn Prediction

This Streamlit app predicts whether a bank customer will churn (leave the bank) based on their demographic and account information using an Artificial Neural Network (ANN).

## Features
- Interactive input form for customer data
- Real-time churn probability prediction
- ANN model trained on 10,000 customer records

## How to Use
1. Enter customer details (Geography, Gender, Age, Credit Score, etc.)
2. Click to get prediction
3. View churn probability and recommendation

## Model Details
- **Algorithm**: Artificial Neural Network (TensorFlow/Keras)
- **Features**: Credit Score, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
- **Accuracy**: ~86% on test set
