"""AI Machine Failure Predictor - Stage 8 improved baseline.

Educational prototype using the UCI AI4I 2020 Predictive Maintenance dataset.
This is not a certified industrial safety system.
"""
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

DATA_FILE = Path(__file__).with_name("ai4i2020 (2).csv")
FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET = "Machine failure"
# Lower threshold catches more failures, but also creates more false alarms.
FAILURE_THRESHOLD = 0.30


def load_data():
    if not DATA_FILE.exists():
        print(f"\nCSV not found beside this script: {DATA_FILE.name}")
        supplied = input("Enter the full path to your CSV file: ").strip().strip('"')
        csv_path = Path(supplied)
    else:
        csv_path = DATA_FILE

    if not csv_path.exists():
        raise FileNotFoundError(
            "Could not find the dataset. Put the CSV in the same folder as main.py "
            "or enter its correct path."
        )

    df = pd.read_csv(csv_path)
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    if df[required].isnull().any().any():
        raise ValueError("Required columns contain missing values. Clean the CSV first.")
    return df


def main():
    print("=" * 58)
    print("       AI MACHINE FAILURE PREDICTOR - STAGE 8")
    print("=" * 58)
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    print(f"\nDataset rows: {len(df)}")
    print(f"Failure cases: {int(y.sum())}")
    print(f"No-failure cases: {int((y == 0).sum())}")
    print("\nUsing only five sensor features; failure-type columns are excluded to avoid leakage.")

    # Separate training, validation and test sets. The validation set is not used
    # to fit the model; the fixed threshold was selected during Stage 8 comparison.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight=None
    )
    model.fit(X_train, y_train)

    # The 0.30 threshold is a recall-oriented choice from a validation comparison.
    val_probabilities = model.predict_proba(X_val)[:, 1]
    val_predictions = (val_probabilities >= FAILURE_THRESHOLD).astype(int)
    print("\nValidation metrics (threshold = 0.30):")
    print(f"Accuracy:  {accuracy_score(y_val, val_predictions):.2%}")
    print(f"Precision: {precision_score(y_val, val_predictions, zero_division=0):.2%}")
    print(f"Recall:    {recall_score(y_val, val_predictions, zero_division=0):.2%}")
    print(f"F1-score:  {f1_score(y_val, val_predictions, zero_division=0):.2%}")

    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= FAILURE_THRESHOLD).astype(int)
    cm = confusion_matrix(y_test, test_predictions)

    print("\nHeld-out test results (threshold = 0.30):")
    print(f"Accuracy:  {accuracy_score(y_test, test_predictions):.2%}")
    print(f"Precision: {precision_score(y_test, test_predictions, zero_division=0):.2%}")
    print(f"Recall:    {recall_score(y_test, test_predictions, zero_division=0):.2%}")
    print(f"F1-score:  {f1_score(y_test, test_predictions, zero_division=0):.2%}")
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, test_predictions, zero_division=0))
    print(f"False alarms: {cm[0, 1]}")
    print(f"Missed failures: {cm[1, 0]}")
    print("\nThreshold trade-off: lowering the threshold may catch more failures but raises false alarms.")
    print("Do not use this educational model as the sole basis for real-world safety decisions.")

    print("\n" + "-" * 58)
    print("Try a sensor reading. Enter the values in the units shown.")
    print("Type q at the Air temperature prompt to quit.")

    while True:
        raw = input("\nAir temperature [K] (e.g. 300): ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            print("Goodbye.")
            break
        try:
            air = float(raw)
            process = float(input("Process temperature [K] (e.g. 310): "))
            rpm = float(input("Rotational speed [rpm] (e.g. 1500): "))
            torque = float(input("Torque [Nm] (e.g. 40): "))
            wear = float(input("Tool wear [min] (e.g. 100): "))
            values = pd.DataFrame([{
                FEATURES[0]: air,
                FEATURES[1]: process,
                FEATURES[2]: rpm,
                FEATURES[3]: torque,
                FEATURES[4]: wear,
            }])
            probability = float(model.predict_proba(values)[0, 1])
            prediction = int(probability >= FAILURE_THRESHOLD)
            print(f"\nEstimated failure probability: {probability:.1%}")
            if prediction:
                print("Prediction: FAILURE RISK FLAGGED - inspect the machine.")
            else:
                print("Prediction: FAILURE RISK NOT FLAGGED by this model.")
            print("This is a model estimate, not a guarantee of machine condition.")
        except ValueError:
            print("Please enter numeric values (or type q to quit).")


if __name__ == "__main__":
    main()