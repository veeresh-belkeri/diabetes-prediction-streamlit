
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# Load the saved model bundle
@st.cache_resource
def load_model_bundle():
    return joblib.load("diabetes_model.joblib")

bundle = load_model_bundle()

# Extract the actual trained pipeline and threshold
model = bundle["model"]
threshold = bundle["threshold"]
FEATURES = bundle["features"]

st.title("🩺 Diabetes Risk Predictor")

st.write(
    "Enter the measurements below to view the model's prediction."
)

st.warning(
    "Educational demonstration only. This is not a medical "
    "diagnosis or a substitute for professional advice."
)

with st.form("diabetes_form"):
    st.subheader("Patient measurements")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "Pregnancies", min_value=0, max_value=20, value=1
        )
        glucose = st.number_input(
            "Glucose (mg/dL)", min_value=0.0,
            max_value=300.0, value=120.0
        )
        blood_pressure = st.number_input(
            "Blood Pressure (mm Hg)", min_value=0.0,
            max_value=200.0, value=70.0
        )
        skin_thickness = st.number_input(
            "Skin Thickness (mm)", min_value=0.0,
            max_value=100.0, value=20.0
        )

    with col2:
        insulin = st.number_input(
            "Insulin (mu U/ml)", min_value=0.0,
            max_value=1000.0, value=80.0
        )
        bmi = st.number_input(
            "BMI", min_value=0.0, max_value=80.0, value=25.0
        )
        dpf = st.number_input(
            "Diabetes Pedigree Function", min_value=0.0,
            max_value=3.0, value=0.5
        )
        age = st.number_input(
            "Age (years)", min_value=1,
            max_value=120, value=30
        )

    submitted = st.form_submit_button(
        "Predict", type="primary"
    )

if submitted:
    input_data = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ]],
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
    )

    # Apply the saved pipeline to raw inputs
    probability = model.predict_proba(input_data)[0, 1]

    # Use the selected threshold
    prediction = int(probability >= threshold)

    st.subheader("Prediction result")

    st.metric(
        "Model-estimated probability",
        f"{probability:.1%}"
    )

    if prediction == 1:
        st.error("The model classified this input as class 1.")
    else:
        st.success("The model classified this input as class 0.")

    st.caption(f"Classification threshold: {threshold:.2f}")

    st.info(
        "This result is for educational purposes only and "
        "does not establish whether someone has diabetes."
    )
