import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import os

# Load the trained model with error handling
MODEL_DIR = 'Models'
try:
    # Try .keras format first (newer), then .h5 (legacy)
    keras_path = os.path.join(MODEL_DIR, 'model.keras')
    h5_path = os.path.join(MODEL_DIR, 'model.h5')
    
    if os.path.exists(keras_path):
        model = tf.keras.models.load_model(keras_path)
    elif os.path.exists(h5_path):
        model = tf.keras.models.load_model(h5_path)
    else:
        st.error(f"Model not found in {MODEL_DIR}/ folder")
        st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.error(f"TensorFlow version: {tf.__version__}")
    st.error("Please ensure model file is compatible with this TF version")
    st.stop()

# Load the encoders and scaler with error handling
try:
    with open(os.path.join(MODEL_DIR, 'label_encoder_gender.pkl'), 'rb') as file:
        label_encoder_gender = pickle.load(file)

    with open(os.path.join(MODEL_DIR, 'onehot_encoder_geo.pkl'), 'rb') as file:
        onehot_encoder_geo = pickle.load(file)

    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as file:
        scaler = pickle.load(file)
except Exception as e:
    st.error(f"Error loading preprocessing files from {MODEL_DIR}/: {e}")
    st.stop()


## streamlit app
st.title('Customer Churn PRediction')

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
