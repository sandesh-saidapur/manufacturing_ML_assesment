# Manufacturing Quality Prediction & Streaming Anomaly Detection

## Overview

This project addresses two parts of the assessment:

1. Predict manufacturing defects from sensor data using an end-to-end machine learning pipeline.
2. Implement a streaming anomaly detector for continuous numerical sensor data.

The dataset contains timestamped manufacturing observations with four sensor features:

- `temperature_c`
- `pressure_bar`
- `vibration_mm_s`
- `humidity_pct`

The target is `defect`.

---

## 1. Dataset and Data Quality

The dataset contains:

- 5,000 observations
- 4 numerical sensor features
- 1 binary target
- observations from January 1 to March 1, 2025
- 2 features containing missing values:
  - temperature: 155 missing values (3.1%)
  - humidity: 107 missing values (2.1%)
- no duplicate rows

The target distribution is:

- No defect: 41.96%
- Defect: 58.04%

Because the target is not extremely imbalanced, accuracy is informative but is not sufficient as the primary metric.

---

## 2. Modeling Approach

### Baseline

The baseline is a shallow Decision Tree classifier with median imputation for numerical features.

The model is intentionally simple and provides a transparent reference point.

### Improved model

The improved model uses `HistGradientBoostingClassifier`.

It was selected because the problem contains nonlinear relationships between sensor measurements and defects, and gradient boosting can capture feature interactions that a simple baseline may miss.

---

## 3. Evaluation Metrics

The main metrics are:

### F1 score

F1 balances precision and recall.

This is useful because both types of errors matter in manufacturing:

- False positives can cause unnecessary inspections or production interventions.
- False negatives can allow defective products to pass through quality control.

### ROC-AUC

ROC-AUC measures the model's ability to discriminate between defective and non-defective observations across classification thresholds.

Accuracy is also reported, but F1 and ROC-AUC are emphasized because they provide more information about classification quality.

---

## 4. Results

### Random stratified split

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Decision Tree baseline | 0.9300 | 0.9775 | 0.9000 | 0.9372 | 0.9619 |
| HistGradientBoosting | **0.9550** | 0.9668 | **0.9552** | **0.9610** | **0.9927** |

The improved model increases F1 from 0.9372 to 0.9610 and ROC-AUC from 0.9619 to 0.9927.

---

## 5. Temporal Validation

A random split can hide temporal changes in a production environment.

Therefore, observations were sorted by timestamp and split at February 15, 2025:

- Training period: January 1 – February 14
- Future holdout: February 15 – March 1

### Temporal results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 0.9048 | 0.9897 | 0.8571 | 0.9187 | 0.9411 |
| HistGradientBoosting | **0.9504** | 0.9738 | **0.9464** | **0.9599** | **0.9923** |

The improved model retains similar performance on the future holdout:

- F1: 0.9610 → 0.9599
- ROC-AUC: 0.9927 → 0.9923

This suggests that the observed temporal shift did not materially reduce discrimination in this particular holdout.

---

## 6. Failure Modes and Biases

### Temporal distribution shift

The target prevalence changes over time.

The defect rate is:

- Training period: 56.48%
- Future period: 62.72%

Feature distributions also shift:

| Feature | PSI |
|---|---:|
| Temperature | 0.1653 |
| Pressure | 0.2751 |
| Vibration | 0.0204 |
| Humidity | 0.0088 |

Temperature and pressure show the largest observed distribution changes.

The mitigation is to use temporal validation and monitor feature distributions and target prevalence in production.

### Sensor availability degradation

The original dataset has relatively low missingness, but increased sensor failure can substantially affect model quality.

A stress test artificially introduced missing values across all sensor features:

| Missingness per sensor | F1 | ROC-AUC |
|---:|---:|---:|
| Normal | 0.9610 | 0.9927 |
| 5% | 0.9424 | 0.9858 |
| 10% | 0.9267 | 0.9792 |
| 20% | 0.8813 | 0.9528 |
| 30% | 0.8493 | 0.9243 |

At 20% simulated missingness, F1 decreases from 0.9610 to 0.8813.

The current pipeline uses median imputation, while production monitoring should track missingness per sensor and alert when sensor availability deviates substantially from historical behavior.

---

## 7. Distribution Shift Monitoring

The experiment calculates Population Stability Index (PSI) between the training period and future period for each sensor.

PSI is used as a monitoring signal rather than as an automatic retraining decision. Appropriate thresholds should be calibrated against historical process variability.

Production monitoring should include:

- feature distribution drift
- sensor missingness
- prediction distribution
- defect prevalence once labels become available
- F1 / recall / precision when delayed ground-truth labels arrive
- model latency and error rates

A sustained change in these signals should trigger investigation before automatically retraining the model.

---

# 8. Streaming Anomaly Detector

The engineering component implements a rolling z-score anomaly detector.

Interface:

```python
detector.update(x) -> bool

---

## 9. Testing

The streaming anomaly detector is covered by 15 automated tests.

The tests cover:

- warm-up behavior
- normal observations
- outlier detection
- missing values
- NaN values
- infinite values
- zero variance
- bounded window size
- reset behavior
- invalid configuration
- rolling mean and variance
- sliding-window statistics
- invalid non-numeric input

Run the test suite with:

```bash
python -m pytest -v


---

## 10. Project Structure
.
├── data/
│   └── manufacturing_quality_dataset.csv
├── scripts/
│   ├── analyze_failure_modes.py
│   ├── benchmark_detector.py
│   ├── run_experiment.py
│   └── test_missingness_robustness.py
├── src/
│   ├── __init__.py
│   ├── anomaly_detector.py
│   ├── data.py
│   ├── evaluation.py
│   └── modeling.py
├── tests/
│   └── test_anomaly_detector.py
├── requirements.txt
├── README.md
└── REPORT.md

---

## 11. Running the Project

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate

- Install dependencies:
    pip install -r requirements.txt

- Run the main experiment:
    python -m scripts.run_experiment

- Run the failure-mode analysis:
    python -m scripts.analyze_failure_modes

- Run the missingness robustness experiment:
    python -m scripts.test_missingness_robustness

- Run the anomaly detector benchmark:
    python -m scripts.benchmark_detector

- Run the tests:
    python -m pytest -v


---

## 12. Limitations

This is a time-boxed assessment implementation rather than a production-ready ML system.

Key limitations include:

- the dataset is relatively small
- only four sensor features are available
- no additional machine or process metadata is available
- hyperparameter tuning was deliberately limited
- no probability calibration analysis was performed
- PSI thresholds were not calibrated against a longer historical baseline
- missingness stress testing uses random missingness rather than realistic sensor-specific failure mechanisms
- production monitoring would require reliable delayed ground-truth labels

These would be addressed before deploying the system to a real manufacturing environment.