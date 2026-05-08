# AI-Based Student Performance Prediction System

This project predicts a student's **GPA (Grade Point Average)** using a Machine Learning model trained on academic, demographic, and behavioral factors.  
A **Streamlit web application** is included to allow users to input student details and instantly receive a GPA prediction

## Project Overview

This system uses a *Random Forest Regression model* trained on a dataset of 2,392 students.  
The features include:

- Age  
- Gender  
- Ethnicity  
- Parental education  
- Study time  
- Absences  
- Tutoring  
- Parental support  
- Extracurricular activities  
- Sports / Music / Volunteering  
- GPA (Target variable)

The project includes:

- Fully functional ML model  
- Full exploratory data analysis (EDA)  
- Visualizations  
- Streamlit user interface  
- A complete project report  

## Folder Structure
student_performance/
│
├── app.py -- Streamlit web app (UI)
├── student_performance.py -- ML model training + EDA
├── gpa_predictor.pkl -- Saved RandomForest model
│
├── data/
│ └── student_performance.csv -- Dataset (add your dataset here)
│
├── Project_Report.pdf -- Complete project documentation
├── requirements.txt -- Required Python libraries
└── README.md -- (This file)

## -------- Machine Learning Model -------

### Algorithm Used:
- RandomForestRegressor (100 estimators)

### Train/Test Split:
- 80% Training
- 20% Testing

### Performance Metrics:

| Metric | Score |
|--------|--------|
| MAE | 0.22 |
| RMSE | 0.33 |
| R² | 0.87 |

Random Forest performed best compared to Linear Regression and Ridge Regression.

##  Visualizations Included 

The following plots are generated in the training script:

- GPA Distribution  
- Study Time vs GPA  
- Correlation Heatmap  
- GPA by Gender  
- GPA by Parental Support  
- Actual vs Predicted GPA  

All visuals are available in -- "Project_Report.pdf"  as well.

## Streamlit Web App  
Located in "app.py".

### User Inputs:
- Age  
- Gender  
- Ethnicity  
- Parental Education  
- Study Time Weekly  
- Absences  
- Tutoring  
- Parental Support  
- Extracurricular Activities  
- Sports / Music / Volunteering  

### Output:
--- Predicted GPA (0.0 – 4.0)  

## How to Run This Project

### 1. Install all required libraries

1.  Open terminal in project folder and run:
  
   pip install -r requirements.txt

2. Run the Machine Learning Training Script
   
   python student_performance.py

3. Run the Streamlit App
 
   streamlit run app.py



--------------Thank you for exploring this project!---------------


