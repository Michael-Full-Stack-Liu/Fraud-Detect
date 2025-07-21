import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# title and description
st.title("credit card fraud detection")
st.write("please input the transaction details below to predict if it is fraudulent or not."
"The model uses V1-V28 and Amount as input features, with V1-V28 defaulting to the average values from the dataset.")

# load model and Scaler
@st.cache_resource
def load_model():
    with open("fraud_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return scaler   

# load default values for V1-V28 and Amount
@st.cache_data
def load_default_values():
    with open("X_mean.json", "r") as f:
        data = json.load(f)  # load the mean values of V1-V28 and Amount
    X = data  # drop Class and Time columns
    return X  # return V1-V28 and mean value of Amount 

# load model, Scaler and default values
model = load_model()
scaler = load_scaler()
default_values = load_default_values()

# input transaction details form
st.subheader("Input Transaction Details")
with st.form("transaction_form"):
    inputs = {}
    # fill V1-V28 with default values
    for i in range(1, 29):
        inputs[f'V{i}'] = default_values[f'V{i}']
    inputs['Amount'] = st.number_input("Transaction Amount", value=default_values['Amount'], min_value=0.0, step=0.01)
    inputs['Hour'] = st.number_input("Hour of the day (0-23)", value=0, min_value=0, max_value=23)
    Is_Night = st.selectbox("Is it night time?(0-6)", options=["Yes", "No"], index=0)
    inputs['Is_Night'] = 1 if Is_Night == "Yes" else 0 #modify yes or no object to 1 or 0
    inputs['Amount_scaled'] = scaler.transform(np.array([[inputs['Amount']]])) [0,0] # scale the Amount using the loaded scaler
    Amount_high = st.selectbox("Is the transaction amount high(>=300)?", options=["Yes", "No"], index=0)  
    inputs['Amount_high'] = 1 if Amount_high == "Yes" else 0  # modify yes or no object to 1 or 0
    # submit button
    submitted = st.form_submit_button("Predict")

    if submitted:
        # prepare input data
        input_data = pd.DataFrame([inputs])
        # make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        # display prediction result
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"Fraudulent transaction detected! (Fraud probability: {prediction_proba[1]:.2%})")
        else:
            st.success(f"Non-fraudulent transaction (Non-fraud probability: {prediction_proba[0]:.2%})")

