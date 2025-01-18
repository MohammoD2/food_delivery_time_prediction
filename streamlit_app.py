import streamlit as st
import pandas as pd
import joblib
import json
import mlflow
from sklearn.pipeline import Pipeline
from scripts.data_clean_utils import perform_data_cleaning

# Set up MLflow Tracking URI
mlflow.set_tracking_uri("https://dagshub.com/MohammoD2/food_delivery_time_prediction.mlflow")

# Page Configuration
st.set_page_config(
    page_title="Food Delivery Time Prediction",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS for styling (inline)
st.markdown(
    """
    <style>
    body {
        font-family: Arial, sans-serif;
        background-color: #f9f9f9;
    }
    .header {
        text-align: center;
        color: #333;
    }
    .header h1 {
        color: #ff5733;
        font-size: 2.5rem;
    }
    .header p {
        font-size: 1.2rem;
        color: #555;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #888;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Preload model and preprocessor
@st.cache_resource
def preload_resources():
    with open("run_information.json") as f:
        run_info = json.load(f)
    model_name = run_info['model_name']
    stage = "staging"
    model_path = f"models:/{model_name}/{stage}"

    # Load the model
    model = mlflow.sklearn.load_model(model_path)

    # Load the preprocessor
    preprocessor = joblib.load("models/preprocessor.joblib")

    # Create a prediction pipeline
    model_pipe = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('regressor', model)
    ])
    return model_pipe

# Load resources once
model_pipe = preload_resources()

# HTML Header and Styling
st.markdown(
    """
    <div class="header">
        <h1>🍔 Food Delivery Time Prediction App</h1>
        <p>Predict delivery times with precision and speed!</p>
    </div>
    <hr>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("Navigation")
st.sidebar.info(
    """
    Use this app to predict food delivery times.
    - Input the required fields
    - Click **Predict** to get the estimated delivery time
    """
)

# Input Form
with st.form("prediction_form"):
    st.markdown("<h2>Enter Delivery Details</h2>", unsafe_allow_html=True)
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

    # Perform data cleaning (ensure this function is efficient)
    cleaned_data = perform_data_cleaning(input_data)

    # Predict
    prediction = model_pipe.predict(cleaned_data)[0]

    st.success(f"Predicted Delivery Time: {prediction:.2f} minutes")

# Footer
st.markdown(
    """
    <div class="footer">
        <p>Made with ❤️ by MohammoD2</p>
    </div>
    """,
    unsafe_allow_html=True,
)


