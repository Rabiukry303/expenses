import pandas as pd
import joblib
import streamlit as st

model = joblib.load("expense_model.pkl")
encoder = joblib.load("expense_encoder.pkl")

salary = st.number_input("enter your salary")
level = st.number_input("enter your level")

if st.button("Predict"):
    sample = pd.DataFrame({
        'salary':[salary],
        'level':[level]
        })
    converted = encoder.transform(sample)
    make_prediction  = model.predict(converted)
    st.success(f"Your predicted expense is: {make_prediction[0]}")