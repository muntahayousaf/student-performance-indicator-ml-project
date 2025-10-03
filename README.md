# Student Math Score Prediction Project 🎯

## 📌 Overview
This project predicts a student's **Math Score** based on other academic features using **machine learning**.  
It demonstrates an **end-to-end workflow**, from data preprocessing to model deployment via **Streamlit**.

**Achieved Model Performance:**  
- **Linear Regression** with **88% accuracy (R²)** on the test set.


## Dataset
The dataset contains student information including:

| Feature | Description |
|---------|-------------|
| gender | Student's gender |
| race/ethnicity | Student's race/ethnicity group |
| parental level of education | Highest education level of the student's parent |
| lunch | Type of lunch (standard / free-reduced) |
| test preparation course | Completed or none |
| reading score | Reading exam score |
| writing score | Writing exam score |
| math score | Math exam score (Target variable) |

**Notes:**  
- Contains both **categorical** and **numerical** features.  

---

## Data Preprocessing

### 1. Categorical Encoding
- Categorical features (`gender`, `race/ethnicity`, `parental level of education`, `lunch`, `test preparation course`) were encoded using **One-Hot Encoding**.  
- `drop='first'` was applied to avoid multicollinearity.

### 2. Feature Scaling
- Numerical features (`reading score`, `writing score`) were scaled using **StandardScaler** to normalize the data.

### 3. Final Features
- Encoded categorical features + scaled numerical features combined into the final dataset for training.

---

## Model Selection
- Multiple regression models were evaluated:  
  - Linear Regression  
  - Lasso Regression  
  - Ridge Regression  
  - Decision Tree Regressor  
  - Random Forest Regressor  
  - K-Neighbors Regressor  
  - XGBoost Regressor  
  - CatBoost Regressor  
  - AdaBoost Regressor  

- Evaluation Metrics: **RMSE, MAE, R²**  
- **Best Model:** Linear Regression with **R² = 0.88** on test data.

---

## Deployment with Streamlit
- The trained model, **OneHotEncoder**, and **StandardScaler** were saved using **pickle**.  
- Users can input student features in the web app and receive **predicted Math Score**.  

**Run the app:**
```bash
streamlit run app.py
