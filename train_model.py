import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from joblib import dump
import json

# Load dataset
df = pd.read_csv("heart_disease_synthetic.csv")

# Rename target column if needed
if "Label" in df.columns:
    df.rename(columns={"Label": "target"}, inplace=True)

# Separate features and labels
X = df.drop("target", axis=1)
y = df["target"]

# Feature Selection: top 8 features
selector = SelectKBest(score_func=f_classif, k=8)
X_selected = selector.fit_transform(X, y)
mask = selector.get_support()
selected_features = X.columns[mask].tolist()

print("✅ Top 8 features used for training:", selected_features)

# Save selected feature names to JSON
with open("features.json", "w") as f:
    json.dump(selected_features, f)

# Create new DataFrame with selected features
X = df[selected_features]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
print(f"🎯 RandomForest Accuracy: {rf_acc:.4f}")
dump(rf_model, "random_forest_model.pkl")

# KNN - find best k
best_k = 1
best_k_acc = 0
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test))
    if acc > best_k_acc:
        best_k_acc = acc
        best_k = k

knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train, y_train)
print(f"🎯 Best KNN (k={best_k}) Accuracy: {best_k_acc:.4f}")
dump(knn_model, "knn_model.pkl")

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_acc = accuracy_score(y_test, log_model.predict(X_test))
print(f"🎯 Logistic Regression Accuracy: {log_acc:.4f}")
dump(log_model, "logistic_model.pkl")

print("✅ Models trained and saved successfully!")
print("📌 Use the features in this exact order:", selected_features)
