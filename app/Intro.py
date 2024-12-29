import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

# Configure page
st.set_page_config(layout="wide", page_title="Employee Attrition Analysis")

# Main Title
st.title("Soft Set Approach for Conflict Analysis and Predictive Analysis")
st.subheader("A Data Science Project on Employee Attrition Classification")

# Section: Project Overview
st.markdown("## 📜 Project Overview")
st.markdown("""
Employee attrition is a critical challenge for organizations. This project explores a novel
approach using **Soft Set Theory** to analyze conflicts and predict employee attrition.

### **Objective:**
- To apply Soft Set Theory to analyze conflicts related to employee satisfaction and turnover.
- To identify key employee attributes that contribute to attrition and conflict within the organization.
- To create a simple platform for HR managers to visualize and analyze conflict factors affecting employee satisfaction and attrition.
""")

# Section: Conflict Flow Graph
st.markdown("## 🔍 Conflict Flow Graph")
st.write("The following conflict flow graph illustrates the relationships and conflicts identified during the analysis:")

# Placeholder for the conflict flow graph
conflict_graph_path = "images/conflict_graph.png" 
st.image(conflict_graph_path, caption="Conflict Flow Graph")

# Section: Models and Results
st.markdown("## 📊 Models and Results")
st.write("Below are the models used in the project and their respective performance metrics:")

# Example data (Replace this with your actual results)
models_data = {
"Model": ["Logistic Regression","ADABoost", "XGBoost", "Random Forest", "Gradient Boosting"],
"Accuracy": [0.79, 0.84, 0.88, 0.86, 0.86],
"F1 Score": [0.81, 0.86, 0.86, 0.84, 0.85],
"AUC_ROC":[0.80, 0.77, 0.80, 0.82, 0.80]
}

# Display results as a DataFrame
results_df = pd.DataFrame(models_data)
st.dataframe(results_df)

# Visualize Results
st.markdown("### Performance Comparison")
fig, ax = plt.subplots(figsize=(10, 6))
results_df.set_index("Model").plot(kind="bar", ax=ax)
plt.title("Model Performance Metrics")
plt.ylabel("Score")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.tight_layout()

# Show the chart in Streamlit
st.pyplot(fig)

# Section: About the Author
st.markdown("## 👤 About the Author")
st.expander("""
Hi, I’m Kong Yan Hao, a third-year Computer Science student at UM with a keen interest 
in **Data Science** and **Soft Set Theory Applications**.
""")