
# AI Machine Failure Predictor
# Stage 3: User input and validation

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

try:
    temperature = float(input("Enter temperature (C): "))
    rpm = float(input("Enter motor speed (RPM): "))
    vibration = float(input("Enter vibration (mm/s): "))

    # Validate the readings
    if not 0 <= temperature <= 150:
        print("Error: Temperature must be 0 to 150 C.")
    elif not 0 < rpm <= 10000:
        print("Error: RPM must be above 0 and at most 10000.")
    elif not 0 <= vibration <= 30:
        print("Error: Vibration must be 0 to 30 mm/s.")
    else:
        result = predict_machine(
            temperature, rpm, vibration
        )

        print("\n---------- RESULT ----------")
        print("Temperature:", temperature, "C")
        print("Motor speed:", rpm, "RPM")
        print("Vibration:", vibration, "mm/s")
        print("Prediction:", result)
        print("----------------------------")

except ValueError:
    print("Error: Please enter numbers only.")


