
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Loading the Dataset
calories = pd.read_csv("data/calories.csv")
exercise_data= pd.read_csv("data/exercise.csv")

# Joining them 
calories_data=pd.concat([exercise_data,calories['Calories']],axis=1)

#Converting Gender from text to num
calories_data.replace({"Gender": {'male': 0, 'female': 1}}, inplace=True)

#Seperate data for model training
X=calories_data.drop(columns=['User_ID','Calories'],axis=1)
Y=calories_data['Calories']

#Seprate the model for training and testing  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape,X_train.shape,X_test.shape)

#Training the RandomForest Regressor Model
model =RandomForestRegressor()
model.fit(X_train,Y_train)

#Streamlit 
st.title("Calories Burnt Prediction")
st.markdown("### Predict the calories burned during exercise based on input parameters.")
st.image("images/calories_logo.jpeg", caption="Stay Fit, Stay Healthy", width=200)

st.markdown("### Input Your Details")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("👤 Gender", ["Male", "Female"])
    age = st.slider("📅 Age", 10, 100, 25)
    height = st.number_input("📏 Height (in cm)", value=170)
    duration = st.number_input("⏱ Duration of Exercise (in minutes)", value=30)

with col2:
    weight = st.number_input("⚖ Weight (in kg)", value=70)
    heart_rate = st.number_input("❤ Heart Rate (bpm)", value=100)
    body_temp = st.number_input("🌡 Body Temperature (°C)", value=37.0)

# Convert gender to numerical
gender = 0 if gender == "Male" else 1

# Button to predict calories
if st.button("Predict"):
    user_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Height': [height],
        'Weight': [weight],
        'Duration': [duration],
        'Heart_Rate': [heart_rate],
        'Body_Temp': [body_temp]
    })

    # Predict using the model
    prediction = model.predict(user_data)
    st.success(f"Predicted Calories Burned: {prediction[0]:.2f}")

# Model Performance Section
st.markdown("---")
st.markdown("### Model Performance Metrics")
with st.expander("View Metrics"):
    # Evaluate the model
    Y_predict = model.predict(X_test)
    mae = mean_absolute_error(Y_test, Y_predict)
    r2 = r2_score(Y_test, Y_predict)

    st.write(f"*Mean Absolute Error (MAE):* {mae:.2f} kcal")
    st.write(f"*R² Score:* {r2 * 100:.2f}%")
