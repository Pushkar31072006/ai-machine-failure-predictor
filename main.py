
# AI Machine Failure Predictor
# Beginner version: no external libraries required

def predict_machine(temperature, rpm, vibration):
    if temperature > 90 or vibration > 8:
        return "Potential failure"
    elif temperature > 80 and vibration > 5:
        return "Potential failure"
    else:
        return "Normal"


print("================================")
print(" AI MACHINE FAILURE PREDICTOR")
print("================================")

temperature = 95
rpm = 1500
vibration = 9.0

print("Temperature:", temperature, "C")
print("Motor speed:", rpm, "RPM")
print("Vibration:", vibration, "mm/s")

result = predict_machine(temperature, rpm, vibration)

print("--------------------------------")
print("Prediction:", result)
print("--------------------------------")

