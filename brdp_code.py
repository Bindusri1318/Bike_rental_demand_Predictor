# ===============================
# Bike Rental Demand Predictor
# ===============================
# This project uses ML (Regression) + Streamlit UI

# -------------------------------
# 1. Install Requirements
# -------------------------------
# pip install pandas numpy scikit-learn streamlit matplotlib seaborn

# -------------------------------
# 2. Import Libraries
# -------------------------------
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# 3. Load Dataset
# -------------------------------
# Use bike sharing dataset (hour.csv or day.csv)
df = pd.read_csv("day.csv")

# -------------------------------
# 4. Data Preprocessing
# -------------------------------
# Drop unnecessary columns

df = df.drop(['instant','dteday','casual','registered'], axis=1)

# Features and Target
X = df.drop('cnt', axis=1)
y = df['cnt']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 5. Model Training
# -------------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 6. Evaluation
# -------------------------------
y_pred = model.predict(X_test)

print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# -------------------------------
# 7. Save Model
# -------------------------------
pickle.dump(model, open('bike_model.pkl', 'wb'))


# ===============================
# STREAMLIT APP
# ===============================

# Save below code in app.py and run: streamlit run app.py

"""
import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('bike_model.pkl', 'rb'))

st.title("🚲 Bike Rental Demand Predictor")

st.write("Enter details to predict bike demand")

# Inputs
season = st.selectbox("Season (1:Spring,2:Summer,3:Fall,4:Winter)", [1,2,3,4])
yr = st.selectbox("Year (0:2011,1:2012)", [0,1])
mnth = st.slider("Month", 1, 12)
holiday = st.selectbox("Holiday", [0,1])
weekday = st.slider("Weekday (0-6)", 0, 6)
workingday = st.selectbox("Working Day", [0,1])
weathersit = st.selectbox("Weather (1:Clear,2:Mist,3:Light Rain,4:Heavy Rain)", [1,2,3,4])
temp = st.slider("Temperature (Normalized)", 0.0, 1.0)
atemp = st.slider("Feels Like Temp", 0.0, 1.0)
hum = st.slider("Humidity", 0.0, 1.0)
windspeed = st.slider("Wind Speed", 0.0, 1.0)

# Prediction
if st.button("Predict"):
    input_data = np.array([[season, yr, mnth, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Bike Rentals: {int(prediction[0])}")
"""

# ===============================
# END OF PROJECT
# ===============================
