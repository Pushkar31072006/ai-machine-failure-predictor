AI Machine Failure Predictor
Educational project using the UCI AI4I 2020 Predictive Maintenance Dataset.

Place the CSV file in the same folder as this script. The script trains a
Random Forest classifier, evaluates it on a held-out test set, and lets you
enter sensor readings for a prediction.

This is a learning demonstration, not a safety-critical industrial system.
"""

from pathlib import Path
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

DATASET_CANDIDATES = [
    "ai4i2020 (2).csv",
    "ai4i2020.csv",
    "ai4i2020_dataset.csv",
]

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET = "Machine failure"


def locate_dataset():
    """Find the dataset beside this script, or ask for a path."""
    folder = Path(__file__).resolve().parent

    for filename in DATASET_CANDIDATES:
        candidate = folder / filename
        if candidate.exists():
            return candidate

    csv_files = sorted(folder.glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]

    print("Could not automatically find the AI4I CSV file.")
    print("Place the dataset beside main.py, or enter its full file path.")
    entered = input("CSV path: ").strip().strip('"')
    candidate = Path(entered).expanduser()
    if candidate.exists() and candidate.is_file():
        return candidate

    print("ERROR: CSV file not found. Check the filename/path and try again.")
    sys.exit(1)


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "The CSV is missing required columns: " + ", ".join(missing)
        )

    # Keep only the sensor readings and target; coerce malformed values.
    df = df[required].apply(pd.to_numeric, errors="coerce").dropna()
    df[TARGET] = df[TARGET].astype(int)

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Machine failure must contain only 0 and 1.")
    if df[TARGET].nunique() != 2:
        raise ValueError("Dataset must contain both failure classes (0 and 1).")

    return df


def train_and_evaluate(df):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("\n========== MODEL EVALUATION ==========")
    print(f"Rows used: {len(df)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Failure cases in full data: {int(y.sum())}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["No failure", "Failure"],
        zero_division=0,
    ))
    print(
        "Reminder: accuracy alone can be misleading because failures are rare. "
        "Check failure-class recall and precision too."
    )
    return model


def get_number(prompt, minimum=None, maximum=None):
    while True:
        raw = input(prompt).strip()
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a numeric value.")
            continue

        if minimum is not None and value < minimum:
            print(f"Value must be at least {minimum}.")
            continue
        if maximum is not None and value > maximum:
            print(f"Value must be no more than {maximum}.")
            continue
        return value


def predict_interactively(model):
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


