import streamlit as st
import pandas as pd
import io
from xgboost import XGBClassifier
from joblib import load
from sklearn.ensemble import GradientBoostingClassifier

st.set_page_config(layout="wide")

# Load models
xgb_model = XGBClassifier()
xgb_model.load_model('models/XGBoost.json')

models = {
    "XGBoost (Recommended)": xgb_model,
    "Random Forest": load('models/RandomForest.pkl'),
    # "ADABoost": load('models/ada_clf_model.pkl'),
    "Gradient Boosting": load('models/Gradient_Boosting.pkl')
}

# App title
st.title("Attrition Prediction with Batch Processing")

# Model selection
st.sidebar.header("Select a Model")
selected_model_name = st.sidebar.selectbox("Choose a Model", list(models.keys()))
selected_model = models[selected_model_name]

# Preprocessing function for batch input
def preprocess_batch(data):
    # Ensure required columns are present
    required_columns = [
        "EmployeeNumber", "JobSatisfaction", "WorkLifeBalance", "EnvironmentSatisfaction",
        "RelationshipSatisfaction", "Age", "MonthlyIncome", "YearsAtCompany", "OverTime",
        "JobInvolvement", "Education", "PerformanceRating"
    ]
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"The input file is missing required columns: {missing_cols}")
    
    # Convert OverTime to binary
    data['OverTime'] = data['OverTime'].apply(lambda x: 1 if x == "Yes" else 0)
    
    # Keep EmployeeNumber separate for final output
    employee_numbers = data['EmployeeNumber']
    
    # Select relevant features for the model
    features = data[[
        "JobSatisfaction", "WorkLifeBalance", "EnvironmentSatisfaction", "RelationshipSatisfaction",
        "Age", "MonthlyIncome", "YearsAtCompany", "OverTime", "JobInvolvement", "Education", "PerformanceRating"
    ]]
    return employee_numbers, features

# File upload for batch prediction
st.header("Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type=["csv"])

if uploaded_file:
    try:
        # Read the uploaded file
        input_data = pd.read_csv(uploaded_file)
        
        # Preprocess the data
        employee_numbers, processed_data = preprocess_batch(input_data)
        
        # Predict attrition with handling for models without feature names
        if isinstance(selected_model, GradientBoostingClassifier):
            predictions = selected_model.predict(processed_data.to_numpy())  # Convert to NumPy array to avoid feature name issue
        else:
            predictions = selected_model.predict(processed_data)
        
        # Add predictions to the results
        results = pd.DataFrame({
            "EmployeeNumber": employee_numbers,
            "Predicted_Attrition": ["Yes" if pred == 1 else "No" for pred in predictions]
        })

        # Display results
        st.write("Batch Prediction Results:")
        st.dataframe(results)

        # Provide download link for the result
        output_csv = results.to_csv(index=False)
        st.download_button(
            label="Download Prediction Results as CSV",
            data=io.StringIO(output_csv).getvalue(),
            file_name="attrition_predictions.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"An error occurred: {e}")
