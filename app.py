import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("gpa_predictor.pkl")

st.title("🎓 Student GPA Predictor")

st.markdown("Enter the student details below:")

# Input fields
age = st.slider("Age", 15, 18, 16)
gender = st.radio("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
ethnicity = st.selectbox("Ethnicity", [0, 1, 2, 3],
                         format_func=lambda x: ["Caucasian", "AfricanAmerican", "Asian", "Other"][x])
parent_ed = st.slider("Parental Education Level (0=none to 4=higher)", 0, 4, 2)
study_time = st.slider("Study Time Weekly (hours)", 0.0, 20.0, 5.0)
absences = st.slider("Number of Absences", 0, 30, 5)
tutoring = st.radio("Tutoring", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
parent_support = st.slider("Parental Support (0=none to 4=very high)", 0, 4, 2)
extra = st.radio("Extracurricular Activities", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
sports = st.radio("Sports", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
music = st.radio("Music", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
volunteering = st.radio("Volunteering", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

# Prediction button
if st.button("🎯 Predict GPA"):
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Ethnicity": ethnicity,
        "ParentalEducation": parent_ed,
        "StudyTimeWeekly": study_time,
        "Absences": absences,
        "Tutoring": tutoring,
        "ParentalSupport": parent_support,
        "Extracurricular": extra,
        "Sports": sports,
        "Music": music,
        "Volunteering": volunteering
    }])

    prediction = model.predict(input_data)[0]
    st.success(f"📚 Predicted GPA: **{prediction:.2f}** (Max: 4.0)")

