import os
import re
import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# ── Universal patch: tolerates unknown config keys from newer Keras versions ──
from keras.engine.base_layer import Layer as _BaseLayer

_orig_base_from_config = _BaseLayer.from_config.__func__

@classmethod
def _tolerant_from_config(cls, config):
    config = dict(config)
    # Known incompatible keys between Keras 3.x and 2.x
    config.pop('quantization_config', None)
    config.pop('optional', None)
    if 'batch_shape' in config:
        config['batch_input_shape'] = config.pop('batch_shape')
    try:
        return _orig_base_from_config(cls, config)
    except TypeError as e:
        # Dynamically strip any remaining unrecognized kwargs
        unknown_keys = re.findall(r"Unrecognized keyword arguments?: \[?'?(\w+)'?\]?", str(e))
        if not unknown_keys:
            unknown_keys = re.findall(r"'(\w+)'", str(e))
        for key in unknown_keys:
            config.pop(key, None)
        return cls(**config)

_BaseLayer.from_config = _tolerant_from_config
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, 'Models', 'model.h5'),
    compile=False
)

with open(os.path.join(BASE_DIR, 'Models', 'label_encoder_gender.pkl'), 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open(os.path.join(BASE_DIR, 'Models', 'onehot_encoder_geo.pkl'), 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open(os.path.join(BASE_DIR, 'Models', 'scaler.pkl'), 'rb') as file:
    scaler = pickle.load(file)

st.title('Customer Churn Prediction')

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

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')