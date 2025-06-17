import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from fpdf import FPDF
from datetime import datetime
import os

# Load models
rf_model = load("random_forest_model.pkl")
knn_model = load("knn_model.pkl")
log_model = load("logistic_model.pkl")

# Page config
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("💓 Heart Disease Prediction App")
st.markdown("Enter your health details to assess your heart disease risk using three ML models.")

# Personal Information
st.header("👤 Patient Information")
name = st.text_input("Patient Name")
email = st.text_input("Email")
location = st.text_input("Location")

# Medical Inputs
st.header("🩺 Medical Inputs")
age = st.slider("Age", 20, 100, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])

# Convert to numeric
sex_val = 1 if sex == "Male" else 0
cp_val = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}[cp]
fbs_val = 1 if fbs == "True" else 0
exang_val = 1 if exang == "Yes" else 0

input_data = pd.DataFrame([[age, sex_val, cp_val, trestbps, chol, fbs_val, thalach, exang_val]],
                          columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'thalach', 'exang'])

# Prediction
if st.button("🚀 Predict and Generate Report"):
    # Predict with models
    rf_pred = rf_model.predict(input_data)[0]
    knn_pred = knn_model.predict(input_data)[0]
    log_pred = log_model.predict(input_data)[0]

    # Results
    results = {
        "Random Forest": "High Risk" if rf_pred == 1 else "Low Risk",
        "KNN": "High Risk" if knn_pred == 1 else "Low Risk",
        "Logistic Regression": "High Risk" if log_pred == 1 else "Low Risk"
    }

    # Final result (can be from RF or majority vote)
    final_risk = "High Risk" if rf_pred == 1 else "Low Risk"

    # Show results
    st.subheader("🔍 Prediction Results")
    for model, result in results.items():
        st.write(f"**{model}:** {result}")
    
    if final_risk == "High Risk":
        st.error("⚠️ You may be at **high risk** of heart disease. Please consult a doctor.")
    else:
        st.success("✅ You are at **low risk** of heart disease. Stay healthy!")

    # Bar chart visualization
    st.subheader("📊 Prediction Comparison")
    chart_data = pd.DataFrame({
        'Model': list(results.keys()),
        'Risk Score (0=Low, 1=High)': [rf_pred, knn_pred, log_pred]
    })
    st.bar_chart(chart_data.set_index("Model"))

    # Health Tips
    st.subheader("💡 Health Tips")
    st.markdown("""
    - 🍎 Eat a fiber-rich, low-fat diet  
    - 🏃‍♂️ Exercise daily for at least 30 minutes  
    - 🚭 Avoid tobacco and limit alcohol  
    - 🧘 Manage stress with yoga or meditation  
    - 🔍 Regularly monitor blood pressure and sugar levels  
    """)

    # Save pie chart image
    labels = ['High Risk', 'Low Risk']
    values = [sum([rf_pred, knn_pred, log_pred]), 3 - sum([rf_pred, knn_pred, log_pred])]
    colors = ['#FF6B6B', '#6BCB77']

    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%', colors=colors)
    chart_img_path = "reports/chart.png"
    os.makedirs("reports", exist_ok=True)
    plt.savefig(chart_img_path)
    plt.close()

    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Heart Disease Prediction Report", ln=True)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Name: {name}", ln=True)
    pdf.cell(0, 10, f"Email: {email}", ln=True)
    pdf.cell(0, 10, f"Location: {location}", ln=True)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Input Data", ln=True)
    pdf.set_font("Arial", size=12)
    for col, val in zip(input_data.columns, input_data.values[0]):
        pdf.cell(0, 10, f"{col}: {val}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Model Predictions", ln=True)
    pdf.set_font("Arial", size=12)
    for model, result in results.items():
        pdf.cell(0, 10, f"{model}: {result}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Health Tips", ln=True)
    tips = [
        "Eat a fiber-rich, low-fat diet",
        "Exercise daily for at least 30 minutes",
        "Avoid tobacco and limit alcohol",
        "Manage stress with yoga or meditation",
        "Monitor blood pressure and sugar levels"
    ]
    pdf.set_font("Arial", size=12)
    for tip in tips:
        pdf.cell(0, 10, f"- {tip}", ln=True)

    # Add pie chart to PDF
    pdf.image(chart_img_path, w=100)

    # Save and offer download
    filename = f"{name.replace(' ', '_')}_Heart_Report.pdf"
    filepath = os.path.join("reports", filename)
    pdf.output(filepath)

    with open(filepath, "rb") as f:
        st.download_button("📥 Download Report as PDF", f, file_name=filename)
    st.success("📄 Report saved and ready to download!")
    st.balloons()
