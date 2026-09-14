# Methodology

## 1. Analytical Unit

The analytical unit is an **inpatient hospital encounter**.

Each row in the modeling dataset represents a completed inpatient encounter with sufficient subsequent observation time to determine whether another inpatient admission begins within 30 days of discharge.

---

## 2. Target Construction

The target variable is `readmitted_30d`.

For each patient:

1. inpatient encounters are sorted chronologically by admission timestamp;
2. the next inpatient admission is identified;
3. the interval between the current discharge and the next inpatient admission is calculated;
4. the current encounter is labeled `readmitted_30d = 1` when the next inpatient admission begins within 0–30 days after discharge;
5. otherwise, the encounter is labeled `readmitted_30d = 0`; and
6. encounters without a complete 30-day follow-up period at the end of the observable dataset are excluded.

Excluding encounters near the end of the observation period reduces right-censoring by avoiding automatic non-readmission labels when a complete 30-day follow-up window is unavailable.

---

## 3. Feature Engineering

The final model uses **18 predictors** grouped into patient, current-encounter, prior-utilization, calendar, and insurance characteristics.

### Patient Characteristics

#### Age at Admission

`age_at_admission` is calculated using the patient's date of birth and the admission date of the current inpatient encounter.

Age is calculated at the time of admission rather than relative to the current date.

Additional patient characteristics include:

- `gender`
- `race`
- `ethnicity`
- `marital`

### Current Encounter Characteristics

#### Length of Stay

`length_of_stay_days` is calculated as:

```text
length_of_stay_days = discharge timestamp - admission timestamp
```

Invalid negative durations are excluded during data preparation.

Other current-encounter characteristics include:

- `base_encounter_cost`
- `total_claim_cost`
- `payer_coverage`
- `procedure_count`

#### Procedure Count

Procedure records are aggregated at the encounter level. The number of procedures associated with each encounter is then joined to the inpatient encounter table as `procedure_count`.

### Prior-Utilization Characteristics

#### Prior Inpatient Visits

`prior_inpatient_visits` represents the number of observed inpatient encounters occurring before the current encounter for the same patient.

#### Prior Readmissions

`prior_readmissions` represents the number of earlier encounters followed by a readmission within 30 days.

The readmission outcome is shifted before cumulative summation so that the current encounter's outcome is not included in its own historical predictor.

#### Days Since Previous Discharge

`days_since_previous_discharge` measures the interval between the previous inpatient discharge and the current admission.

The previous discharge timestamp is created by shifting discharge history within each patient.

For the first observed inpatient encounter for a patient, no previous inpatient discharge is available. These observations are identified using `missing_prior_history`.

### Calendar Characteristics

Calendar features are derived from the admission timestamp:

- `admission_month`
- `admission_dayofweek`
- `admission_weekend`

`admission_dayofweek` uses Monday = 0 through Sunday = 6. `admission_weekend` equals 1 for Saturday or Sunday admissions and 0 otherwise.

### Insurance Characteristic

`payer_name` is obtained by joining the encounter payer identifier to the payer table.

---

## 4. Leakage Controls

Several controls are used to reduce target leakage:

- inpatient encounters are ordered chronologically within each patient;
- future admissions are used only to construct the target;
- historical utilization variables use information from earlier encounters only;
- the current encounter's readmission outcome is excluded from its own `prior_readmissions` value;
- encounters without a complete 30-day follow-up period are excluded;
- training and testing are separated chronologically; and
- preprocessing transformations are fitted on the training data and then applied to the test data.

---

## 5. Prediction Point

Several predictors describe the completed index hospitalization, including:

- `length_of_stay_days`
- `total_claim_cost`
- `payer_coverage`
- `procedure_count`

These variables may not be fully known when a patient is first admitted.

The current model is therefore interpreted as a **discharge-time or post-encounter analytical model**, rather than a strictly admission-time prediction model.

An admission-time model would require restricting predictors to information available before or at the beginning of the inpatient encounter.

---

## 6. Validation Design

A chronological **80/20 train/test split** is used.

Earlier encounters form the training set, while later encounters form the held-out test set.

This approach preserves the temporal structure of the longitudinal encounter data and more closely represents evaluation on future encounters than a random row split.

The resulting split contains:

| Partition | Encounters | Readmission Rate |
|---|---:|---:|
| Training | 905 | 37.13% |
| Test | 227 | 32.16% |

The majority-class baseline accuracy in the test set is approximately **67.84%**.

---

## 7. Preprocessing

### Numeric Variables

Numeric variables are processed using:

- median imputation; and
- standard scaling.

### Categorical Variables

Categorical variables are processed using:

- most-frequent imputation; and
- one-hot encoding with unknown categories ignored.

Preprocessing is incorporated into scikit-learn pipelines so transformations learned from the training data are applied consistently to the held-out test data.

---

## 8. Algorithms

Two classification algorithms are compared.

### Logistic Regression

Logistic Regression provides an interpretable linear baseline.

Balanced class weights are used to account for differences in the frequency of readmission and non-readmission outcomes.

### Random Forest

Random Forest provides a nonlinear comparison model capable of representing more complex relationships and interactions among predictors.

The model uses balanced class weights, constrained tree depth, a minimum leaf size, and a fixed random seed.

---

## 9. Evaluation Metrics

Model performance is evaluated using:

- accuracy;
- precision;
- recall;
- F1 score;
- ROC-AUC;
- PR-AUC; and
- confusion matrix.

ROC and Precision-Recall curves are also used for visual evaluation.

Accuracy is not interpreted alone. Precision, recall, PR-AUC, and ROC-AUC are considered together when comparing model performance.

Using PR-AUC as the primary comparison metric and ROC-AUC as the secondary metric, Random Forest is selected as the stronger of the two candidate models on the held-out chronological test set.

---

## 10. Model Interpretation

Permutation importance is calculated using the held-out test data to assess how strongly the fitted model depends on each original predictor.

The analysis identifies several leading predictors, including:

1. `base_encounter_cost`
2. `total_claim_cost`
3. `days_since_previous_discharge`
4. `procedure_count`
5. `prior_readmissions`
6. `payer_coverage`

Permutation importance represents **predictive association rather than causality**. A variable with high importance should not be interpreted as causing readmission.

---

## 11. Risk Bands

Predicted probabilities from the selected model are grouped into three analytical categories:

| Probability | Risk Category |
|---|---|
| 0.00–0.30 | Low |
| >0.30–0.60 | Medium |
| >0.60–1.00 | High |

These categories are descriptive analytical groupings and are **not clinically validated thresholds**.

---

## 12. Reproducibility

Randomized algorithms use a fixed `random_state = 42` where applicable.

Raw source files are not modified. Intermediate and final analytical outputs are written separately to `data/processed/`.

Reusable data-preparation and modeling logic is maintained in the `src/` directory, while the notebooks document the analytical workflow and results.

---

## 13. Methodological Limitations

- The source dataset is synthetic.
- The number of patients is relatively small for clinical predictive modeling.
- Observed readmissions are limited to encounters represented in the source data.
- Important clinical variables may not be available.
- Several current-encounter predictors are fully known only during or by the end of hospitalization.
- Demographic associations are descriptive and should not be interpreted as causal relationships.
- Model performance on this dataset should not be generalized to real hospital populations.
- The model is an analytical demonstration and is not intended for clinical decision-making.
