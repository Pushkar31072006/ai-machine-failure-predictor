
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Machine data: temperature, RPM, vibration
X = np.array([
    [50, 1000, 1],
    [60, 1500, 2],
    [70, 1800, 3],
    [95, 1500, 9],
    [100, 2000, 10],
    [85, 1200, 8],
    [55, 1400, 1.5],
    [105, 2500, 11]
])

# 0 = Normal, 1 = Potential failure
y = np.array([0, 0, 0, 1, 1, 1, 0, 1])

# Train the machine learning model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X, y)

# Predict a new machine's condition
machine = np.array([[95, 1500, 9]])
prediction = model.predict(machine)[0]

print("AI MACHINE FAILURE PREDICTOR")
print("Prediction:",
      "Potential failure" if prediction == 1 else "Normal")
