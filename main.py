
import numpy as np
from sklearn.ensemble import RandomForestClassifier

print("Scikit-learn is working!")
print("NumPy version:", np.__version__)

X = np.array([
    [50, 1000, 1],
    [60, 1500, 2],
    [70, 1800, 3],
    [95, 1500, 9],
    [100, 2000, 10],
    [85, 1200, 8],
    [55, 1400, 1.5],
    [105, 2500, 11],
    [65, 1600, 2],
    [92, 1800, 9.5],
    [75, 1700, 4],
    [98, 2200, 10]
])

# 0 = Normal, 1 = Potential failure
y = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1])

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

machine = np.array([[95, 1500, 9]])
prediction = model.predict(machine)[0]

print("Training completed successfully!")
print("Test readings:", machine[0])
print(
    "Prediction:",
    "Potential failure" if prediction == 1 else "Normal"
)


