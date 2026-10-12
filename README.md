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
- The uploaded CSV contains 10,000 rows and 14 columns, with 339 `Machine failure = 1` records and no missing values in the dataset.
- The dataset is synthetic and intended for research/education; it does not represent every real industrial machine.

## Model evaluation (verified run)

The uploaded CSV was evaluated using the five sensor features listed above, a stratified 80/20 train/test split, and `RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)`.

- Rows used: **10,000**
- Training rows: **8,000**
- Test rows: **2,000**
- Test-set accuracy: **98.50%**
- Failure precision: **89.58%** (43 of 48 predicted failures were actual failures)
- Failure recall: **63.24%** (43 of 68 actual failures were detected)
- Failure F1-score: **74.14%** (rounded)

Confusion matrix (rows = actual class, columns = predicted class):

| | Predicted no failure | Predicted failure |
|---|---:|---:|
| Actual no failure | 1,927 | 5 |
| Actual failure | 25 | 43 |

**Interpretation:** The model correctly flagged 43 of the 68 failure cases in the test set, but missed 25. Because failures are uncommon, accuracy alone is not enough to judge the model. Recall, precision, and false negatives are important too.

The results above correspond to the dataset and model configuration described here. They may differ if the dataset, feature set, split, random seed, or library versions change.

## Files

- `main.py` — trains, evaluates, and runs interactive predictions.
- `requirements.txt` — Python packages required.
- `ai4i2020 (2).csv` — dataset file (or use another CSV filename supported by `main.py`).

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

The program will search for the dataset in the same folder. If it cannot find it automatically, it asks for a path.

## Limitations and safety

This is a student learning project, not a certified predictive-maintenance product. A prediction is not a guarantee. Do not use it as the sole basis for operating or maintaining real equipment. Real deployment would require domain-expert review, validation on representative real-machine data, monitoring, and appropriate safety procedures.

## Credits

Dataset: AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository. See the dataset page above for original attribution and reuse terms