import streamlit as st
import pandas as pd
import os
from xgboost import XGBClassifier
from joblib import load

# Define the path to the models directory
models_dir = "models"

# Check if the directory exists
if not os.path.exists(models_dir):
    raise FileNotFoundError(f"The models directory '{models_dir}' does not exist.")

# Load XGBoost model
xgb_model = XGBClassifier()
xgb_model.load_model('models/XGBoost.json')

models = {
    "XGBoost": xgb_model,
    "Random Forest": load('models/RandomForest.pkl'),
    # "ADABoost": load('models/ada_clf_model.pkl'),
    "Gradient Boosting": load('models/Gradient_Boosting.pkl')
}
model_names = ["XGBoost", "Random Forest", "ADABoost", "Gradient Boosting"]

# App Title and Header
st.set_page_config(layout="wide", page_title="Attrition Prediction")
st.title("Employee Attrition Prediction")
st.subheader("Analyze employee attrition using machine learning models")

# Model Selection (Toggle Switch)
st.markdown("### Select a Model")
model_options = ["XGBoost", "Random Forest", "ADABoost", "Gradient Boosting"]
selected_model_index = st.radio("Choose a Model", range(len(model_options)), format_func=lambda x: model_options[x], horizontal=True)
selected_model_name = model_options[selected_model_index]
selected_model = models[selected_model_name]

# Input Features in Main Content
st.markdown("## Input Features")

job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
work_life_balance = st.slider("Work Life Balance (1-4)", 1, 4, 3)
environment_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
age = st.number_input("Age", min_value=18, max_value=65, value=30)
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=5000)
years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
overtime = st.selectbox("OverTime (Yes/No)", ["Yes", "No"])
relationship_satisfaction = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
job_involvement = st.slider("Job Involvement (1-4)", 1, 4, 3)
education = st.slider("Education (1-4)", 1, 4, 3)
performance_rating = st.slider("Performance Rating (1-4)", 1, 4, 3)

# Preprocessing function
def preprocess_input(data):
    data['OverTime'] = data['OverTime'].apply(lambda x: 1 if x == "Yes" else 0)
    columns = [
        "JobSatisfaction", "WorkLifeBalance", "EnvironmentSatisfaction", "RelationshipSatisfaction",
        "Age", "MonthlyIncome", "YearsAtCompany", "OverTime", "JobInvolvement", "Education", "PerformanceRating"
    ]
    return data[columns]

# Prepare input data
input_data = pd.DataFrame({
    "JobSatisfaction": [job_satisfaction],
    "WorkLifeBalance": [work_life_balance],
    "EnvironmentSatisfaction": [environment_satisfaction],
    "Age": [age],
    "MonthlyIncome": [monthly_income],
    "YearsAtCompany": [years_at_company],
    "OverTime": [overtime],
    "RelationshipSatisfaction": [relationship_satisfaction],
    "JobInvolvement": [job_involvement],
    "Education": [education],
    "PerformanceRating": [performance_rating]
})

processed_data = preprocess_input(input_data)

# Prediction and Results
st.markdown("## Prediction Results")
if st.button("Predict Attrition"):
    try:
        # Convert to NumPy array if model was trained without feature names
        if selected_model_name in ["Random Forest", "ADABoost", "Gradient Boosting"]:
            prediction = selected_model.predict(processed_data.to_numpy())[0]
        else:
            prediction = selected_model.predict(processed_data)[0]
        
        result = "Yes" if prediction == 1 else "No"
        st.metric(label="Predicted Attrition", value=result)
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Display Input Summary
st.markdown("### Input Summary")
st.write(input_data)
