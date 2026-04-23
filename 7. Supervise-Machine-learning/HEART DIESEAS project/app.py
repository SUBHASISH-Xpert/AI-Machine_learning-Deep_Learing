import streamlit as st
import pandas as pd
import joblib

model = joblib.load('logistic Regression_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_Columns = joblib.load('columns.pkl')

st.title('Heart Disease Prediction')
st.markdown('Enter the following details to predict the likelihood of heart disease:')
age = st.slider('Age', 20, 80, 40)
sex = st.selectbox('SEX',['M','F'])
chest_pain_type = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'])
resting_bp = st.number_input('Resting Blood Pressure (mm Hg)', min_value=80, max_value=200, value=120)
cholesterol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=400, value=200)
fasting_blood_sugar = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['Yes', 'No'])
rest_ecg = st.selectbox('Resting Electrocardiographic Results', ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
max_heart_rate = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)
exercise_induced_angina = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
oldpeak = st.number_input('Oldpeak (ST depression induced by exercise)', min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])

if st.button('Predict'):
    raw_input ={
        'age': age,
        'sex': 1 if sex == 'M' else 0,
        'chest_pain_type': chest_pain_type,
        'resting_bp': resting_bp,
        'cholesterol': cholesterol,
        'fasting_blood_sugar': 1 if fasting_blood_sugar == 'Yes' else 0,
        'rest_ecg': rest_ecg,
        'max_heart_rate': max_heart_rate,
        'exercise_induced_angina': 1 if exercise_induced_angina == 'Yes' else 0,
        'oldpeak': oldpeak,
        'slope': slope
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_Columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_Columns]
    scled_input = scaler.transform(input_df)
    prediction = model.predict(scled_input)[0]

    if prediction == 1:
        st.error('High likelihood of heart disease. Please consult a doctor.')
    else:   
         st.success('Low likelihood of heart disease. Keep up the healthy lifestyle!')
