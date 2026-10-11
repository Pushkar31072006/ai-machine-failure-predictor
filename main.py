t_interactively(model):
    print("\n========== ENTER SENSOR READINGS ==========")
    print("Enter the current machine sensor values.")
    print("Temperature values are in Kelvin; tool wear is in minutes.\n")

    air_temp = get_number("Air temperature [K]: ", 250, 400)
    process_temp = get_number("Process temperature [K]: ", 250, 450)
    rpm = get_number("Rotational speed [rpm]: ", 0, 10000)
    torque = get_number("Torque [Nm]: ", 0, 500)
    tool_wear = get_number("Tool wear [min]: ", 0, 1000)

    sample = pd.DataFrame(
        [[air_temp, process_temp, rpm, torque, tool_wear]],
        columns=FEATURES,
    )
    prediction = int(model.predict(sample)[0])

    print("\n========== PREDICTION ==========")
    if prediction == 1:
        print("Prediction: POTENTIAL MACHINE FAILURE")
        print("Action: Inspect the machine and verify sensor readings.")
    else:
        print("Prediction: NO FAILURE PREDICTED")
        print("Action: Continue normal monitoring and follow maintenance rules.")

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(sample)[0]
        failure_probability = float(probability[list(model.classes_).index(1)])
        print(f"Model-estimated failure probability: {failure_probability:.1%}")

    print(
        "\nImportant: This prediction is not a guarantee. Do not use this "
        "educational model as the sole basis for real-world safety decisions."
    )


def main():
    csv_path = locate_dataset()
    print(f"Loading dataset: {csv_path.name}")
    df = load_dataset(csv_path)
    print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

    model = train_and_evaluate(df)

    while True:
        predict_interactively(model)
        again = input("\nPredict another reading? (y/n): ").strip().lower()
        if again not in {"y", "yes"}:
            print("Finished. Thank you for trying the AI Machine Failure Predictor.")
            break


if __name__ == "__main__":
    main()



