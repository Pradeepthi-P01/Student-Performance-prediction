# Student Performance Prediction using Random Forest

#----------------IMPORTING OF LIBRARIES --------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load the dataset
df = pd.read_csv('student_performance\Data\Student_performance_data _.csv')
# df = pd.read_csv('Data/Student_performance_data_.csv')


# Step 2: Drop unnecessary columns
df.drop(columns=["StudentID", "GradeClass"], inplace=True)

# Step 3: Data preprocessing
print("Checking for missing values:")
print(df.isnull().sum())

# ----------------- Step 4: EDA - Correlation ---------------------------

# ------------- 1. GPA Distribution ----------
plt.figure(figsize=(8, 5))
sns.histplot(df['GPA'], bins=30, kde=True, color='skyblue')
plt.title("Distribution of Student GPA")
plt.xlabel("GPA")
plt.ylabel("Number of Students")
plt.show()

# ---------- 2. Study Time vs GPA ----------
plt.figure(figsize=(8, 5))
sns.scatterplot(x='StudyTimeWeekly', y='GPA', data=df, color='green')
plt.title("Study Time vs GPA")
plt.xlabel("Study Time Weekly (Hours)")
plt.ylabel("GPA")
plt.show()

# ---------- 3. Correlation Heatmap ----------
plt.figure(figsize=(10, 6))
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

# ---------- 4. GPA by Gender ----------
plt.figure(figsize=(7, 5))
sns.boxplot(x='Gender', y='GPA', data=df, hue='Gender', palette='pastel', legend=False)
plt.title("GPA Distribution by Gender (0 = Female, 1 = Male)")
plt.xlabel("Gender")
plt.ylabel("GPA")
plt.show()

# ---------- 5. GPA by Parental Support ----------
plt.figure(figsize=(7, 5))
sns.boxplot(x='ParentalSupport', y='GPA', data=df, hue='ParentalSupport', palette='Set2', legend=False)
plt.title("GPA vs Parental Support")
plt.xlabel("Parental Support (0 = No, 1 = Yes)")
plt.ylabel("GPA")
plt.show()


X = df.drop("GPA", axis=1)
y = df["GPA"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)

import joblib
joblib.dump(pipeline, 'gpa_predictor.pkl')

y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n Evaluation Metrics:")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([0, 4], [0, 4], 'r--')
plt.xlabel("Actual GPA")
plt.ylabel("Predicted GPA")
plt.title("Actual vs Predicted GPA")
plt.grid(True)
plt.show()

print("\n Try Predicting GPA with Your Own Data")

user_data = {
    "Age": int(input("Enter Age (15-18yrs): ")),
    "Gender": int(input("Enter Gender (1=Male, 0=Female): ")),
    "Ethnicity": int(input("Enter Ethnicity Code (e.g., 0-caucasian,1-AfricanAmerican, 2-Asian, 3-other): ")),
    "ParentalEducation": int(input("Enter Parental Education Level (0=none, 1=high school, 2=some college, 3=bachelors, 4=higher ): ")),
    "StudyTimeWeekly": float(input("Enter Weekly Study Time (in hours (0-20): ")),
    "Absences": int(input("Enter Number of Absences(in yr (0-30)): ")),
    "Tutoring": int(input("Tutoring (1=Yes, 0=No): ")),
    "ParentalSupport": int(input("Parental Support (0=none, 1=low, 2=moderate, 3=high, 4=very high): ")),
    "Extracurricular": int(input("Extracurricular (1=Yes, 0=No): ")),
    "Sports": int(input("Sports (1=Yes, 0=No): ")),
    "Music": int(input("Music (1=Yes, 0=No): ")),
    "Volunteering": int(input("Volunteering (1=Yes, 0=No): "))
}

user_df = pd.DataFrame([user_data])

predicted_gpa = pipeline.predict(user_df)[0]
print(f"\n Predicted GPA (max. of 4) : {predicted_gpa:.2f}")
