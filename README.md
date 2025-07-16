# 💓 Heart Disease Predictor

This is a machine learning project that predicts the risk of heart disease using patient health data. The app includes a Streamlit web interface and uses multiple models trained on a synthetic heart disease dataset.
<img width="1920" height="3197" alt="chrome-capture-2025-6-17 (1)" src="https://github.com/user-attachments/assets/0ff8fd98-b263-4b17-8aa1-c427673f9ad6" />

---

## 📊 Models Used

- **Random Forest Classifier**
- **K-Nearest Neighbors (KNN)** – Auto-tuned for best `k`
- **Logistic Regression**

All models are trained using the top 8 features selected through ANOVA F-test (`SelectKBest`).

---

## 📁 Project Structure

```

heart-disease-predictor/
├── app.py                      # Streamlit frontend app
├── train\_models.py            # Script to train and save models
├── features.json              # List of top 8 features used
├── random\_forest\_model.pkl    # Trained RF model
├── knn\_model.pkl              # Trained KNN model
├── logistic\_model.pkl         # Trained Logistic Regression model
├── heart\_disease\_synthetic.csv  # Dataset used
├── requirements.txt           # Dependencies
└── reports/                   # Generated PDF reports

````

---

## 🚀 How to Run

1. **Clone the repo**
```bash
git clone https://github.com/MrCoss/heart-disease-predictor.git
cd heart-disease-predictor
````

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the app**

```bash
streamlit run app.py
```

---

## 🧠 Features Used for Prediction

The model uses the following top 8 features:

```
- age
- sex
- cp
- trestbps
- chol
- fbs
- thalach
- exang
```

(Stored in `features.json`)

---

## 📝 Report Generation

After prediction, a personalized PDF report is automatically generated with:

* Patient info
* Model predictions (RF, KNN, Logistic)
* Health tips
* Date and location

---

## 📦 Dependencies

* `streamlit`
* `pandas`, `numpy`
* `scikit-learn`
* `fpdf`
* `joblib`

Install via:

```bash
pip install -r requirements.txt
```

---

## 📬 Author

**Costas Pinto (MrCoss)**
Feel free to fork, use, and improve! Contributions welcome 💡

