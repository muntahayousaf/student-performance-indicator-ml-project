import streamlit as st
import pandas as pd
import numpy as np
import pickle

model= pickle.load(open("C:/Users/admin/OneDrive/Documents/GitHub/student-performance-indicator-ml-project/models/my_model.pkl" , 'rb'))

def main():
    st.title("Student Math Score Prediction")

    # User inputs
    gender = st.selectbox("Gender", ["male", "female"])
    ethnicity = st.selectbox("Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
    parental_education = st.selectbox("Parental Level of Education", [
                                    "some high school", "high school", "some college", "associate's degree",
                                    "bachelor's degree", "master's degree"])
    lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
    prep_course = st.selectbox("Test Preparation Course", ["none", "completed"])
    reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=72)
    writing_score = st.number_input("Writing Score", min_value=0, max_value=100, value=74)
    
    new_data = pd.DataFrame({
    "reading score": [72],
    "writing score": [74],
    "gender": ["female"],
    "race/ethnicity": ["group B"],
    "parental level of education": ["bachelor's degree"],
    "lunch": ["standard"],
    "test preparation course": ["none"]
})

    if st.button('Pridict'):
        makeprediction = model.predict([[gender , ethnicity , parental_education , 
                                         lunch, prep_course , reading_score , writing_score]])
        output = round(makeprediction[0], 2)
        st.success(f"You can get Math score: {output}")


if __name__ == "__main__":
    main()



