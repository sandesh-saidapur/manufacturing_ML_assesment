# Assessment Reflection

## Prioritizations and Intentional Trade-offs

I prioritized the complete and reproducible implementation over tuning. The implementation includes data validation, missing value handling, baseline, enhanced gradient boosting, complimentary metrics, temporal validation, distribution shift analysis, and robustness test.

For engineering side, I selected the streaming anomaly detector as it is aligned with the sensor-based manufacturing use case. For this part of the implementation, I considered bounded memory, O(1) update complexity, warmup behavior, data validation and automated testing.

In light of the time constraint, I consciously decided to avoid feature engineering, hyperparameters search, ensemble of models, or production infrastructure layer.

## What I Would Harden in the Next 2–3 Days

First of all, I would improve the validation of the data and the monitoring of sensor ranges, timestamps, missingness, and schema changes. After that, I would assess probability calibration and classification thresholds taking into account the business cost of false positives and negatives.

For the anomaly detector, I would assess its behavior in case of sensor failures and regime change. I would also add logging, configuration management and integration tests for the inference pipeline.

## Risks and Mitigations

The biggest modeling risk is the distribution shift. The data has been changing its target class distribution and the distribution of sensors over time. This problem is mitigated with a temporal holdout in addition to a random split and calculation of feature-level PSI.

The second risk is sensor availability. While currently the missingness is low, the simulation showed that higher missingness leads to degradation of performance. The mitigation is the monitoring of missingness per each sensor and treating increases of missingness as a signal of data quality problem.

The last risk is a small size of the dataset and limited process context in the data. Thus, the performance in validation can not be treated as a guarantee of production performance. For a proper assessment of the model, additional historical production data and delay in ground truth monitoring would be required.