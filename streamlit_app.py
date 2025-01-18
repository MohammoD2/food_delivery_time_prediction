import streamlit as st
import pandas as pd
import joblib
import json
import mlflow
from sklearn.pipeline import Pipeline
from scripts.data_clean_utils import perform_data_cleaning

# Set up MLflow Tracking URI
mlflow.set_tracking_uri("https://dagshub.com/MohammoD2/food_delivery_time_prediction.mlflow")

# Load model information
def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
    return run_info

# Load the model transformer
def load_transformer(transformer_path):
    transformer = joblib.load(transformer_path)
    return transformer

# Load the pre-trained model
model_name = load_model_information("run_information.json")['model_name']
stage = "staging"
model_path = f"models:/{model_name}/{stage}"
model = mlflow.sklearn.load_model(model_path)

# Load the preprocessor
preprocessor = load_transformer("models/preprocessor.joblib")

# Create a prediction pipeline
model_pipe = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('regressor', model)
])

# Streamlit App
st.title("Swiggy Food Delivery Time Prediction")

st.write("This application predicts the delivery time for Swiggy orders.")

# Input Form
with st.form("prediction_form"):
    ID = st.text_input("ID", value="1")
    Delivery_person_ID = st.text_input("Delivery Person ID", value="1001")
    Delivery_person_Age = st.text_input("Delivery Person Age", value="30")
    Delivery_person_Ratings = st.text_input("Delivery Person Ratings", value="4.5")
    Restaurant_latitude = st.number_input("Restaurant Latitude", value=12.9716)
    Restaurant_longitude = st.number_input("Restaurant Longitude", value=77.5946)
    Delivery_location_latitude = st.number_input("Delivery Location Latitude", value=12.9352)
    Delivery_location_longitude = st.number_input("Delivery Location Longitude", value=77.6245)
    Order_Date = st.text_input("Order Date", value="2025-01-18")
    Time_Orderd = st.text_input("Time Ordered", value="14:30")
    Time_Order_picked = st.text_input("Time Order Picked", value="14:45")
    Weatherconditions = st.selectbox("Weather Conditions", options=["Sunny", "Rainy", "Cloudy"])
    Road_traffic_density = st.selectbox("Road Traffic Density", options=["Low", "Medium", "High", "Jam"])
    Vehicle_condition = st.slider("Vehicle Condition", min_value=1, max_value=10, value=5)
    Type_of_order = st.selectbox("Type of Order", options=["Food", "Groceries"])
    Type_of_vehicle = st.selectbox("Type of Vehicle", options=["Bike", "Car"])
    multiple_deliveries = st.text_input("Multiple Deliveries", value="2")
    Festival = st.selectbox("Festival", options=["Yes", "No"])
    City = st.selectbox("City", options=["Urban", "Semi-Urban", "Rural"])

    # Submit Button
    submitted = st.form_submit_button("Predict")

# Handle Form Submission
if submitted:
    # Convert form inputs into a DataFrame
    input_data = pd.DataFrame({
        'ID': [ID],
        'Delivery_person_ID': [Delivery_person_ID],
        'Delivery_person_Age': [Delivery_person_Age],
        'Delivery_person_Ratings': [Delivery_person_Ratings],
        'Restaurant_latitude': [Restaurant_latitude],
        'Restaurant_longitude': [Restaurant_longitude],
        'Delivery_location_latitude': [Delivery_location_latitude],
        'Delivery_location_longitude': [Delivery_location_longitude],
        'Order_Date': [Order_Date],
        'Time_Orderd': [Time_Orderd],
        'Time_Order_picked': [Time_Order_picked],
        'Weatherconditions': [Weatherconditions],
        'Road_traffic_density': [Road_traffic_density],
        'Vehicle_condition': [Vehicle_condition],
        'Type_of_order': [Type_of_order],
        'Type_of_vehicle': [Type_of_vehicle],
        'multiple_deliveries': [multiple_deliveries],
        'Festival': [Festival],
        'City': [City]
    })

    # Perform data cleaning
    cleaned_data = perform_data_cleaning(input_data)

    # Make prediction
    prediction = model_pipe.predict(cleaned_data)[0]

    st.success(f"Predicted Delivery Time: {prediction:.2f} minutes")

