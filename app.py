# Import necessary libraries
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

#Load the Encoders and Scaler
with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Set the title of the app
st.title('Customer Churn Prediction App')

# Take user input
credit_score = st.number_input('Credit Score', min_value=0, max_value=850, value=600)
geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
gender = st.radio('Gender',['Male','Female'])
Age = st.number_input('Age', min_value=18, max_value=100, value=35)
Tenure = st.number_input('Tenure', min_value=0, max_value=10, value=5)
Balance = st.number_input('Balance', min_value=0, max_value=200000, value=50000)
NumOfProducts = st.number_input('Number of Products', min_value=1, max_value=4, value=2)
HasCrCard = st.selectbox('Has Credit Card', ['Yes', 'No'])
is_active_member = st.radio('Is Active Member', ['Yes', 'No'])
EstimatedSalary = st.number_input('Estimated Salary', min_value=0, value=50000)

# Convert user input to a DataFrame
input_data = {
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': Age,
    'Tenure': Tenure,
    'Balance': Balance,
    'NumOfProducts': NumOfProducts,
    'HasCrCard': HasCrCard == 'Yes',
    'IsActiveMember': is_active_member == 'Yes',
    'EstimatedSalary': EstimatedSalary
}

input_data_df = pd.DataFrame(input_data, index=[0])
# Preprocess the input data

# Encode the Geography column
geo_encoded = onehot_encoder.transform([[input_data['Geography']]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder.get_feature_names_out())

input_data_df = pd.concat([input_data_df,geo_encoded_df],axis=1)
input_data_df.drop('Geography',axis=1,inplace=True)

# Encode the Gender column
input_data_df['Gender'] = label_encoder.fit_transform([input_data['Gender']])

st.write(input_data_df)

# Scale the data
input_data_scaled = scaler.transform(input_data_df)

# Model prediction
model_probability = model.predict(input_data_scaled)

st.write("Model Probability: ", model_probability[0][0])

if model_probability[0][0] > 0.5:
    st.write("The customer is likely to leave the bank.")
else:   
    st.write("The customer is likely to stay with the bank.")
