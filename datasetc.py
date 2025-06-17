import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
n_samples = 500

# Generate synthetic features
age = np.random.randint(29, 77, n_samples)
sex = np.random.randint(0, 2, n_samples)
cp = np.random.randint(0, 4, n_samples)
trestbps = np.random.randint(94, 200, n_samples)
chol = np.random.randint(126, 564, n_samples)
fbs = np.random.randint(0, 2, n_samples)
thalach = np.random.randint(71, 202, n_samples)
exang = np.random.randint(0, 2, n_samples)

# Basic logic for target: high BP, low thalach, or chest pain = risk
target = (
    (trestbps > 140).astype(int) +
    (chol > 240).astype(int) +
    (thalach < 120).astype(int) +
    (cp > 1).astype(int) +
    (age > 55).astype(int)
)
# Label as disease if 3 or more risk conditions met
target = (target >= 3).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'age': age,
    'sex': sex,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'thalach': thalach,
    'exang': exang,
    'target': target
})

# Save to CSV
df.to_csv("heart_disease_synthetic.csv", index=False)
print("✅ Synthetic heart disease dataset created and saved.")
