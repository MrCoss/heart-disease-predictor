#  Heart Disease Risk Predictor
An interactive machine learning application built with **Streamlit** to predict the risk of heart disease based on patient health data. The app leverages a multi-model approach, including **Random Forest, KNN, and Logistic Regression**, trained on the most statistically significant features.

-----

## Table of Contents

  - [1. Project Context & Objective](https://www.google.com/search?q=%231-project-context--objective)
  - [2. The Machine Learning Pipeline](https://www.google.com/search?q=%232-the-machine-learning-pipeline)
  - [3. Core Features & Functionality](https://www.google.com/search?q=%233-core-features--functionality)
  - [4. Project Structure Explained](https://www.google.com/search?q=%234-project-structure-explained)
  - [5. Technical Stack](https://www.google.com/search?q=%235-technical-stack)
  - [6. Local Setup & Usage Guide](https://www.google.com/search?q=%236-local-setup--usage-guide)
  - [7. Author & License](https://www.google.com/search?q=%237-author--license)

-----

## 1\. Project Context & Objective

Heart disease is a leading cause of mortality worldwide. Early detection and risk assessment are vital for preventative care and improving patient outcomes. Machine learning offers a powerful tool to analyze complex health data and identify individuals at risk.

The objective of this project is to build an accessible and intuitive application that demonstrates how machine learning can be applied for preliminary heart disease risk assessment. The tool allows users to input key health metrics and receive an instant prediction, serving as a practical example of deploying multiple classification models in a user-friendly web interface.

-----

## 2\. The Machine Learning Pipeline

The project implements a robust and well-structured machine learning workflow from feature selection to model deployment.

### Step 1: Feature Selection

  - To build efficient and effective models, a feature selection process was employed first.
  - The **ANOVA F-test**, implemented via Scikit-learn's `SelectKBest`, was used to statistically score the features and identify the **top 8 most influential predictors** of heart disease. This reduces model complexity and focuses on the most relevant data.

### Step 2: Data Preprocessing

  - The dataset was split into a training set (80%) and a testing set (20%).
  - **Feature Scaling** was applied using `StandardScaler` to normalize the data. This is crucial for distance-based algorithms like KNN and for the convergence of Logistic Regression.

### Step 3: Multi-Model Training (`train_models.py`)

  - Three distinct classification models were trained on the selected features:
    1.  **Random Forest:** An ensemble model known for its high accuracy and robustness.
    2.  **K-Nearest Neighbors (KNN):** A distance-based classifier. The optimal number of neighbors (`k`) was auto-tuned for best performance.
    3.  **Logistic Regression:** A reliable and interpretable linear model that serves as a strong baseline.

### Step 4: Model Persistence

  - All three trained models, along with the feature scaler and the list of selected feature names, are serialized and saved to disk using `joblib` and `json`. This allows the Streamlit app to load and use them for inference without retraining.

### Step 5: Report Generation

  - After a prediction is made, the application uses the `fpdf` library to dynamically generate a personalized PDF report summarizing the patient's inputs, the model's predictions, and general health advice.

-----

## 3\. Core Features & Functionality

  - **Interactive Risk Assessment:** A clean sidebar with sliders and dropdowns allows users to easily input patient health data.
  - **Multi-Model Prediction:** The app provides risk predictions from three different ML models, offering a more comprehensive diagnostic perspective.
  - **Automated Feature Selection:** Models are built using only the top 8 most statistically significant features, demonstrating a key best practice in machine learning.
  - **Personalized PDF Report Generation:** A standout feature that automatically creates a professional, shareable PDF report with user inputs, model outputs, and health tips.
  - **Modular and Reproducible:** The training logic is separated from the application logic, making the project easy to understand, maintain, and reproduce.

-----

## 4\. Project Structure Explained

The repository is organized logically for clarity and ease of use.

```
heart-disease-predictor/
├── app.py                      # The main Streamlit application script for the frontend UI.
├── train_models.py             # Script to run the full training pipeline and save models.
├── features.json               # A JSON file listing the top 8 features used by the models.
├── random_forest_model.pkl     # The serialized, trained Random Forest model.
├── knn_model.pkl               # The serialized, trained K-Nearest Neighbors model.
├── logistic_model.pkl          # The serialized, trained Logistic Regression model.
├── heart_disease_synthetic.csv # The dataset used for training and testing.
├── requirements.txt            # A list of all Python dependencies.
└── reports/                    # Directory where generated PDF reports are saved.
```

-----
<img width="1920" height="3197" alt="chrome-capture-2025-6-17 (1)" src="https://github.com/user-attachments/assets/0ff8fd98-b263-4b17-8aa1-c427673f9ad6" />


## 5\. Technical Stack

  - **Core Language:** Python
  - **Web Framework:** Streamlit
  - **Machine Learning:** Scikit-learn
  - **Data Manipulation:** Pandas, NumPy
  - **PDF Generation:** FPDF
  - **Model Persistence:** Joblib

-----

## 6\. Local Setup & Usage Guide

To run this application on your local machine, please follow these steps.

### Step 1: Clone the Repository

```bash
git clone https://github.com/MrCoss/heart-disease-predictor.git
cd heart-disease-predictor
```

### Step 2: Create and Activate a Virtual Environment (Recommended)

```bash
# Create the environment
python -m venv venv

# Activate on Windows
.\venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train the Models (Optional)

The repository includes pre-trained models. However, if you wish to retrain them using the `train_models.py` script, you can run:

```bash
python train_models.py
```

### Step 5: Launch the Streamlit Application

```bash
streamlit run app.py
```

Your web browser will automatically open with the running application.

-----

## 7\. Author & License

  - **Author:** Costas Pinto ([MrCoss](https://github.com/MrCoss))
  - **License:** This project is open-source and available for educational and personal use. Contributions are welcome\!
