import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Title
st.title("🏍️ Bike Rental Demand Predictor")
st.markdown("### Predict bike rental demand using weather conditions")

# Load dataset
data = pd.read_csv("day.csv")   # make sure file name matches

# Show dataset
st.subheader("Dataset Preview")
st.write(data.head())

# Select features and target
X = data[['temp', 'hum', 'windspeed']]
y = data['cnt']

# Train model
model = LinearRegression()
model.fit(X, y)

# Sidebar inputs
st.sidebar.header("Enter Weather Conditions")

temp = st.sidebar.slider("Temperature (Normalized)", 0.0, 1.0, 0.3)
hum = st.sidebar.slider("Humidity (Normalized)", 0.0, 1.0, 0.5)
windspeed = st.sidebar.slider("Wind Speed (Normalized)", 0.0, 1.0, 0.2)

# Prediction
if st.button("Predict"):
    prediction = model.predict([[temp, hum, windspeed]])
    st.success(f"🚴 Predicted Bike Rentals: {int(prediction[0])}")

# Visualization
st.subheader("Bike Rental Trend")
st.line_chart(data['cnt'])

# Extra info
st.markdown("### 📊 About")
st.write("This app uses Linear Regression to predict bike rental demand based on weather conditions like temperature, humidity, and wind speed.")