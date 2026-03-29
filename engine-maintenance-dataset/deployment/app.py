import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Load model
model_path = hf_hub_download(
    repo_id="ashuPandey/vehicle_predictive_maintenance",
    filename="maintainance_prediction_v1.joblib"
)

model = joblib.load(model_path)

st.title("Predictive Maintenance System")

# Inputs
rpm = st.number_input("Engine rpm")
oil = st.number_input("Lub oil pressure")
fuel = st.number_input("Fuel pressure")
coolant_p = st.number_input("Coolant pressure")
oil_temp = st.number_input("lub oil temp")
coolant_temp = st.number_input("Coolant temp")

if st.button("Predict"):

    df = pd.DataFrame([[
        rpm,
        oil,
        fuel,
        coolant_p,
        oil_temp,
        coolant_temp
    ]], columns=[
        "Engine rpm",
        "Lub oil pressure",
        "Fuel pressure",
        "Coolant pressure",
        "lub oil temp",
        "Coolant temp"
    ])

    pred = model.predict(df)

    if pred[0] == 1:
        st.error("Engine Fault Detected")
    else:
        st.success("Engine is Normal")
