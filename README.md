# AI Machine Failure Predictor

An educational machine-learning project that predicts whether a machine may fail using sensor readings from the **AI4I 2020 Predictive Maintenance Dataset**.

## What it does

- Loads the AI4I CSV dataset.
- Uses five sensor features: air temperature, process temperature, rotational speed, torque, and tool wear.
- Trains a Random Forest classifier.
- Evaluates it on a stratified 80/20 train/test split.
- Prints accuracy, a confusion matrix, and a classification report.
- Accepts sensor values and produces a prediction.

## Dataset

- **Source:** UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset
- **Dataset page:** https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- **License:** CC BY 4.0 (credit the original dataset authors/source when reusing it)
- The dataset contains 10,000 rows and 14 columns. The `Machine failure` target contains 339 failure records.
- The dataset is synthetic and designed for research/education; it does not represent every real industrial machine.

## Files

- `main.py` — trains, evaluates, and runs interactive predictions.
- `requirements.txt` — Python packages required.
- `ai4i2020 (2).csv` — the dataset file (or use another CSV filename listed in `main.py`).

## Run locally

Use Python 3.10 or newer.

1. Put `main.py`, `requirements.txt`, and the dataset CSV in the same folder.
2. Install packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Run:

   ```bash
   python main.py
   ```

If the CSV has a different name, the program will search for a single CSV in the same folder or ask you to enter the path.

## Evaluation notes

The model evaluates on a held-out test set using a fixed random seed (`random_state=42`) so results can be reproduced with the same dataset and library versions. Since machine failures are uncommon, **accuracy alone is not enough**. Review failure-class precision, recall, and the confusion matrix, especially false negatives (failures the model misses).

Do not copy evaluation numbers into a report until you have run the code and checked the output in your environment. Results can vary with package versions or dataset changes.

## Limitations and safety

This is a student learning project, not a certified predictive-maintenance product. A prediction is not a guarantee. Do not use it as the sole basis for operating or maintaining real equipment. A real deployment would require domain-expert review, validation on representative real-machine data, monitoring, and safety procedures.

## Credits

Dataset: AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository. See the dataset page above for original attribution and license details.
