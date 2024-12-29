import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
import mpld3

# Set page configuration
st.set_page_config(layout="wide", page_title="streData Exploration Tool")

# Sidebar for uploading dataset
# Header
st.header("Data Exploration and Analysis")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV file)", type="csv")

if uploaded_file is not None:
    st.session_state['data'] = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully!")

    # Display Dataset Overview
    st.write("### Dataset Overview")
    st.write(st.session_state['data'].head())

    # Check for missing attributes
    expected_attributes = [
        "JobSatisfaction", "WorkLifeBalance", 
        "EnvironmentSatisfaction", "Age", "MonthlyIncome",
        "YearsAtCompany", "OverTime"
    ]
    missing_attributes = [attr for attr in expected_attributes if attr not in st.session_state['data'].columns]
    st.write("### Attribute Validation")
    if missing_attributes:
        st.warning(f"The following attributes are missing: {', '.join(missing_attributes)}")
    else:
        st.success("All expected attributes are present.")

    # Allow users to select attributes for analysis
    st.write("### Select Attributes for Analysis")
    selected_attributes = st.multiselect(
        "Choose attributes to include in the analysis:",
        st.session_state['data'].columns.tolist(),
        default=[attr for attr in expected_attributes if attr in st.session_state['data'].columns]
    )
    st.session_state['selected_attributes'] = selected_attributes
    if not selected_attributes:
        st.warning("Please select at least one attribute.")
    else:
        # Filtered Dataset Overview
        st.write("### Filtered Dataset")
        filtered_data = st.session_state['data'][selected_attributes]
        st.write(filtered_data.head())

        # Summary Statistics
        st.write("### Summary Statistics")
        st.write(filtered_data.describe())

        # Null Value Counts
        st.write("### Null Value Counts")
        st.write(filtered_data.isnull().sum())

        # Quick Visualizations
        st.write("### Quick Visualizations")
        chart_type = st.selectbox("Choose a chart type", ["Bar Chart", "Line Chart", "Scatter Plot"])

        fig, ax = plt.subplots()

        if chart_type == "Bar Chart":
            x = st.selectbox("X-axis", selected_attributes)
            y = st.selectbox("Y-axis", selected_attributes)
            filtered_data.groupby(x)[y].mean().plot(kind='bar', ax=ax)
            ax.set_title(f"Bar Chart of {y} by {x}")
            ax.set_xlabel(x)
            ax.set_ylabel(y)
        elif chart_type == "Line Chart":
            column = st.selectbox("Column", selected_attributes)
            filtered_data[column].plot(kind='line', ax=ax)
            ax.set_title(f"Line Chart of {column}")
            ax.set_xlabel("Index")
            ax.set_ylabel(column)
        elif chart_type == "Scatter Plot":
            x = st.selectbox("X-axis", selected_attributes)
            y = st.selectbox("Y-axis", selected_attributes)
            ax.scatter(filtered_data[x], filtered_data[y])
            ax.set_title(f"Scatter Plot of {y} vs {x}")
            ax.set_xlabel(x)
            ax.set_ylabel(y)

        fig_html = mpld3.fig_to_html(fig)
        components.html(fig_html, height= 500)

        # Detailed Data Profiling
        st.write("### Detailed Data Profiling")
        if st.button("Generate Data Profiling Report"):
            with st.spinner("Generating report..."):
                profile_report = ProfileReport(filtered_data, title="Pandas Profiling Report", explorative=True)
                st.session_state['profile_report_html'] = profile_report.to_html()
                st.session_state['profile_report'] = profile_report
                st.success("Report generated!")
        if 'profile_report_html' in st.session_state:
            components.html(st.session_state['profile_report_html'], height=1000, scrolling=True)

        # Export Options
        st.write("### Export Options")
        if 'profile_report_html' in st.session_state:
            st.download_button(
                "Download Profiling Report",
                data=st.session_state['profile_report_html'],
                file_name="profiling_report.html",
                mime="text/html"
            )
        if 'filtered_data' in locals():
            st.download_button(
                "Download Filtered Dataset",
                data=filtered_data.to_csv(index=False).encode('utf-8'),
                file_name="filtered_dataset.csv",
                mime="text/csv"
            )
