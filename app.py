import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# -----------------------------
# Load and prepare data
# -----------------------------
df = pd.read_csv("StudentsPerformance.csv")

# Encode categorical columns
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop(['math score'], axis=1)
y = np.where(df['math score'] >= 60, 1, 0)

# Train model
model = LogisticRegression()
model.fit(X, y)

# -----------------------------
# Streamlit App
# -----------------------------
st.title("🎓 Student Performance Predictor")
st.write("Predict if a student will pass based on their data")

# Collect user input
gender = st.selectbox("Gender", ["male", "female"])
race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parent_edu = st.selectbox("Parental Education Level", [
    "some high school", "high school", "some college", "associate's degree",
    "bachelor's degree", "master's degree"
])
lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
prep_course = st.selectbox("Test Preparation Course", ["none", "completed"])
reading_score = st.slider("Reading Score", 0, 100, 50)
writing_score = st.slider("Writing Score", 0, 100, 50)

# Create a DataFrame for input
input_data = pd.DataFrame({
    'gender': [gender],
    'race/ethnicity': [race],
    'parental level of education': [parent_edu],
    'lunch': [lunch],
    'test preparation course': [prep_course],
    'reading score': [reading_score],
    'writing score': [writing_score]
})

# Encode input using same label encoders
for col in input_data.columns:
    if input_data[col].dtype == 'object':
        input_data[col] = le.fit_transform(input_data[col])

# Predict
prediction = model.predict(input_data)

# Display result
if st.button("Predict"):
    if prediction[0] == 1:
        st.success("✅ The student is likely to PASS.")
    else:
        st.error("❌ The student is likely to FAIL.")
