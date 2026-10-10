
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Training examples: temperature, RPM, vibration
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

print("AI MACHINE FAILURE PREDICTOR")
print("Model training completed!")

try:
    temperature = float(input("Temperature (C): "))
    rpm = float(input("Motor speed (RPM): "))
    vibration = float(input("Vibration (mm/s): "))

    if not 0 <= temperature <= 150:
        print("Error: Temperature must be 0 to 150 C.")
    elif not 0 < rpm <= 10000:
        print("Error: RPM must be above 0 and at most 10000.")
    elif not 0 <= vibration <= 30:
        print("Error: Vibration must be 0 to 30 mm/s.")
    else:
        readings = np.array([[temperature, rpm, vibration]])
        prediction = model.predict(readings)[0]

        print("\n--- RESULT ---")
        print("Temperature:", temperature, "C")
        print("RPM:", rpm)
        print("Vibration:", vibration, "mm/s")
        print(
            "Prediction:",
            "Potential failure" if prediction == 1 else "Normal"
        )

except ValueError:
    print("Error: Please enter numbers only.")



