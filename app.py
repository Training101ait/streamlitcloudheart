import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and preprocessing artifacts
# Ensure these files exist in the same folder as the app
model = joblib.load("best_heart_disease_model.joblib")
preprocessor = joblib.load("preprocessor.joblib")
selector = joblib.load("feature_selector.joblib")

def predict_risk(patient_data):
    """
    Given a dictionary of patient data, this function transforms the input
    using the preprocessor and feature selector, and then predicts the risk
    using the trained model.
    """
    # Create DataFrame from input data
    df = pd.DataFrame([patient_data])
    # Preprocess the data
    processed = preprocessor.transform(df)
    if hasattr(processed, 'toarray'):
        processed = processed.toarray()
    # Apply feature selection
    selected = selector.transform(processed)
    # Make prediction using the model
    if hasattr(model, 'predict_proba'):
        pred = model.predict_proba(selected)[0][1]
    else:
        pred = model.predict(selected)[0]
    risk_score = float(pred) * 100
    return risk_score

# App title and description
st.title("Heart Disease Prediction App")
st.write(
    """
    This interactive app uses a DenseNet Neural Network ensemble model to predict 
    the risk of heart disease based on patient data. Please input the patient details 
    on the side and click **Predict Heart Disease Risk** to see the result.
    """
)

# Expandable explanation for each column in the dataset
with st.expander("Data Column Explanations"):
    st.markdown(
    """
    **Age:** Age of the patient in years.

    **Sex:** Biological sex of the patient. Options: *Male*, *Female*.

    **Chest Pain Type:** Type of chest pain experienced. Options:
    - *Typical Angina*
    - *Atypical Angina*
    - *Non-anginal Pain*
    - *Asymptomatic*

    **Resting Blood Pressure:** Blood pressure measured at rest (in mm Hg).

    **Serum Cholesterol:** Level of cholesterol in the blood (in mg/dl).

    **Fasting Blood Sugar:** Indicates whether fasting blood sugar is greater than 120 mg/dl.
    - Options: *"> 120 mg/dl"* or *"<= 120 mg/dl"*

    **Resting ECG:** Results of the resting electrocardiogram. Options:
    - *Normal*
    - *ST-T wave abnormality*
    - *Left ventricular hypertrophy*

    **Max Heart Rate:** Maximum heart rate achieved during an exercise test.

    **Exercise Induced Angina:** Whether the patient experiences angina during exercise.
    - Options: *Yes*, *No*

    **ST Depression:** Depression in the ST segment during exercise compared to rest.

    **ST Slope:** The slope of the peak exercise ST segment. Options:
    - *Upsloping*
    - *Flat*
    - *Downsloping*

    **Number of Major Vessels:** Number of major vessels (0-3) colored by fluoroscopy.

    **Thalassemia:** A blood disorder indicating defects in hemoglobin. Options:
    - *Normal*
    - *Fixed defect*
    - *Reversible defect*
    """
    )

st.header("Enter Patient Data")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
    cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fasting_bs = st.selectbox("Fasting Blood Sugar", options=["> 120 mg/dl", "<= 120 mg/dl"])
    resting_ecg = st.selectbox("Resting ECG", options=["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])

with col2:
    max_hr = st.number_input("Max Heart Rate", min_value=30, max_value=250, value=150)
    exercise_angina = st.selectbox("Exercise Induced Angina", options=["Yes", "No"])
    st_depression = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    st_slope = st.selectbox("ST Slope", options=["Upsloping", "Flat", "Downsloping"])
    num_vessels = st.selectbox("Number of Major Vessels", options=["0", "1", "2", "3"])
    thalassemia = st.selectbox("Thalassemia", options=["Normal", "Fixed defect", "Reversible defect"])

# When the user clicks the predict button, create a patient data dictionary,
# compute the risk score and determine the risk category
if st.button("Predict Heart Disease Risk"):
    patient_data = {
        "Age": age,
        "Sex": sex,
        "Chest Pain Type": chest_pain,
        "Resting Blood Pressure": resting_bp,
        "Serum Cholesterol": cholesterol,
        "Fasting Blood Sugar": fasting_bs,
        "Resting ECG": resting_ecg,
        "Max Heart Rate": max_hr,
        "Exercise Induced Angina": exercise_angina,
        "ST Depression": st_depression,
        "ST Slope": st_slope,
        "Number of Major Vessels": num_vessels,
        "Thalassemia": thalassemia
    }

    risk_score = predict_risk(patient_data)
    # Determine risk category based on the risk score
    if risk_score < 20:
        risk_category = "Low Risk"
    elif risk_score < 50:
        risk_category = "Moderate Risk"
    else:
        risk_category = "High Risk"

    st.subheader("Prediction Result")
    st.write(f"**Predicted Risk Score:** {risk_score:.1f}%")
    st.write(f"**Risk Category:** {risk_category}")
