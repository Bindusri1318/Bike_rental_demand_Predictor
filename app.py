import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Page config
st.set_page_config(page_title="Bike Predictor", page_icon="🏍️", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    h1 {color: #ff4b4b;}
    h2 {color: #1f77b4;}
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🏍️ Bike Rental Demand Predictor")
st.markdown("### 🚴 Predict bike demand using weather conditions")

# Load dataset
data = pd.read_csv("day.csv")

# -------------------------------
# 🧹 DATA CLEANING
# -------------------------------

# Remove duplicates
data = data.drop_duplicates()

# Handle missing values
data = data.dropna()

# Optional: Reset index
data = data.reset_index(drop=True)

st.success("✅ Data cleaned successfully")

st.write("🔍 Missing values in each column:")
st.write(data.isnull().sum())

# Show dataset
col1, col2 = st.columns(2)

with col1:
    st.subheader("📂 Dataset Preview")
    st.dataframe(data.head())

    st.download_button(
        label="📥 Download Dataset",
        data=data.to_csv(index=False),
        file_name="bike_data.csv",
        mime="text/csv"
    )

# Train model
X = data[['temp', 'hum', 'windspeed']]
y = data['cnt']

model = LinearRegression()
model.fit(X, y)

# Sidebar inputs
st.sidebar.header("🌦️ Enter Weather Conditions")

temp = st.sidebar.slider("🌡️ Temperature", 0.0, 1.0, 0.3)
hum = st.sidebar.slider("💧 Humidity", 0.0, 1.0, 0.5)
windspeed = st.sidebar.slider("🌬️ Wind Speed", 0.0, 1.0, 0.2)

season = st.sidebar.selectbox("🌸 Select Season", ["Spring", "Summer", "Fall", "Winter"])

# Prediction
with col2:
    st.subheader("🔮 Prediction")

    if st.button("🚀 Predict Now"):
        prediction = model.predict([[temp, hum, windspeed]])
        value = int(prediction[0])

        st.success(f"🚴 Predicted Bike Rentals: {value}")

        # Fake real vs fraud split
        fraud = int(value * 0.1)
        real = value - fraud

        colA, colB, colC = st.columns(3)
        colA.metric("📊 Total Rentals", value)
        colB.metric("✅ Real Rentals", real)
        colC.metric("⚠️ Fraud/Noise", fraud)

# ================== COLORFUL GRAPHS ==================

sns.set_style("darkgrid")

st.subheader("📈 Colorful Trends & Analysis")

col3, col4 = st.columns(2)

# Line Chart
with col3:
    st.markdown("#### 📊 Bike Rental Trend")
    fig, ax = plt.subplots()
    ax.plot(data['cnt'], color='purple', linewidth=2)
    ax.set_title("Bike Rentals Over Time", color='blue')
    ax.set_xlabel("Days")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# Scatter Plot
with col4:
    st.markdown("#### 🌡️ Temperature vs Rentals")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=data['temp'], y=data['cnt'], color='red')
    ax2.set_title("Temp vs Rentals", color='green')
    st.pyplot(fig2)

# Bar Chart (Monthly)
st.markdown("#### 📅 Monthly Average Rentals")

data['mnth'] = pd.to_datetime(data['dteday']).dt.month
monthly_avg = data.groupby('mnth')['cnt'].mean()

fig3, ax3 = plt.subplots()
monthly_avg.plot(kind='bar', color='orange', ax=ax3)
ax3.set_title("Monthly Avg Rentals", color='brown')
ax3.set_xlabel("Month")
ax3.set_ylabel("Avg Count")
st.pyplot(fig3)

# Pie Chart
st.markdown("#### 🥧 Real vs Fraud Rentals")

total = int(data['cnt'].mean())
fraud = int(total * 0.1)
real = total - fraud

fig4, ax4 = plt.subplots()
ax4.pie(
    [real, fraud],
    labels=['Real', 'Fraud'],
    colors=['green', 'red'],
    autopct='%1.1f%%',
    startangle=90
)
ax4.set_title("Real vs Fraud Distribution")
st.pyplot(fig4)

# ================== EXTRA METRICS ==================

st.subheader("📊 Additional Insights")

avg = int(data['cnt'].mean())
max_val = int(data['cnt'].max())
min_val = int(data['cnt'].min())

col5, col6, col7 = st.columns(3)
col5.metric("📌 Average Rentals", avg)
col6.metric("📈 Max Rentals", max_val)
col7.metric("📉 Min Rentals", min_val)

# Footer
st.markdown("---")
st.markdown("✨ Built with Streamlit | 🎯 Simple & Colorful ML Project 🚀")