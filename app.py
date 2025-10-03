import streamlit as st
import pandas as pd
import numpy as np
import pickle

with open("linear_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("onehot_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

cat_features = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course"
]

num_features = ["reading score", "writing score"]

st.title("📊Student Performance Prediction")

st.write("Enter student details to predict the student's Math Score::")

# Numeric Inputs
reading_score = st.number_input("Reading Score", min_value=0.0, max_value=100.0, value=50.0)
writing_score = st.number_input("Writing Score", min_value=0.0, max_value=100.0, value=50.0)

# Categorical Inputs
gender = st.selectbox("Gender", ["male", "female"])
race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parent_edu = st.selectbox("Parental Level of Education", ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"])
lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])

# Create DataFrame
input_data = pd.DataFrame({
    "gender": [gender],
    "race/ethnicity": [race],
    "parental level of education": [parent_edu],
    "lunch": [lunch],
    "test preparation course": [test_prep],
    "reading score": [reading_score],
    "writing score": [writing_score]
})

# Separate categorical and numerical
X_cat = encoder.transform(input_data[cat_features]).toarray()
X_cat_df = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(cat_features))

X_num_scaled = scaler.transform(input_data[num_features])
X_num_df = pd.DataFrame(X_num_scaled, columns=num_features)

# Combine
input_processed = pd.concat([X_num_df, X_cat_df], axis=1)

if st.button("Predict Math Score"):
    pred = model.predict(input_processed)[0]
    st.success(f"Predicted Math Score: {pred:.2f}")