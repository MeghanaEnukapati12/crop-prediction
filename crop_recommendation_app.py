import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# Title
st.title('Crop Recommendation System')
st.write('Enter the soil and environmental parameters to get crop recommendations')

# Load and prepare data
@st.cache_data
def load_data():
    data = pd.read_csv('Crop_recommendation.csv')
    X = data.drop('label', axis=1)
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    model = lgb.LGBMClassifier()
    model.fit(X_train, y_train)
    return model

model = load_data()

# Create input form
col1, col2 = st.columns(2)

with col1:
    n = st.number_input('Nitrogen (N)', min_value=0, max_value=140)
    p = st.number_input('Phosphorus (P)', min_value=0, max_value=145)
    k = st.number_input('Potassium (K)', min_value=0, max_value=205)
    temperature = st.number_input('Temperature (°C)', min_value=0.0, max_value=50.0)

with col2:
    humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0)
    ph = st.number_input('pH', min_value=0.0, max_value=14.0)
    rainfall = st.number_input('Rainfall (mm)', min_value=0.0, max_value=300.0)

# Make prediction
if st.button('Get Crop Recommendation'):
    input_data = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)[0]
    
    st.success(f'The recommended crop for your soil conditions is: *{prediction}*')
    
    # Additional information
    st.info("""
    Note: This recommendation is based on:
    - N-P-K values
    - Temperature
    - Humidity
    - pH level
    - Rainfall
    """)
