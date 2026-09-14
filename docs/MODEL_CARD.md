# Model Card

## Model Purpose

Estimate the probability that a completed inpatient encounter is followed by another inpatient admission within 30 days of discharge.

The model is developed using synthetic hospital data and is intended as an analytical machine learning demonstration.

---

## Intended Use

This project is intended for:

- analytical experimentation;
- development of a reproducible machine learning workflow;
- comparison of linear and nonlinear classification approaches;
- exploration of longitudinal hospital utilization patterns; and
- evaluation of 30-day readmission prediction using synthetic encounter data.

---

## Not Intended For

The model is not intended for:

- diagnosis;
- treatment selection;
- patient triage;
- clinical decision-making;
- insurance eligibility or coverage decisions; or
- deployment as a hospital production model without independent validation using appropriate real-world data.

---

## Data

The project uses the **Maven Analytics Hospital Patient Records** dataset.

The source data is synthetic and includes:

- patient demographics;
- hospital encounters;
- procedures;
- payer information; and
- organization information.

The final analytical dataset is restricted to eligible inpatient encounters with sufficient follow-up time to determine the 30-day readmission outcome.

---

## Target

The target variable is:

`readmitted_30d`

The positive class is defined as:

```text
The next inpatient admission begins within 0–30 days after discharge from the current inpatient encounter.
```

Encounters without a complete 30-day follow-up period at the end of the observable dataset are excluded.

---

## Prediction Point

The final model includes several characteristics of the completed index encounter, including length of stay, procedure count, total claim cost, and payer coverage.

Because these variables may not be fully known when the patient is first admitted, the model is best interpreted as a **discharge-time or post-encounter analytical model** rather than a strictly admission-time prediction model.

---

## Predictors

The final model uses **18 predictors** across five groups:

### Patient Characteristics

- `age_at_admission`
- `gender`
- `race`
- `ethnicity`
- `marital`

### Current Encounter Characteristics

- `length_of_stay_days`
- `base_encounter_cost`
- `total_claim_cost`
- `payer_coverage`
- `procedure_count`

### Prior-Utilization Characteristics

- `prior_inpatient_visits`
- `prior_readmissions`
- `days_since_previous_discharge`
- `missing_prior_history`

### Calendar Characteristics

- `admission_month`
- `admission_dayofweek`
- `admission_weekend`

### Insurance Characteristic

- `payer_name`

---

## Candidate Models

Two classification models are compared:

- Logistic Regression
- Random Forest

Both models use balanced class weights.

---

## Validation

A chronological **80/20 holdout split** is used.

| Partition | Encounters | Readmission Rate |
|---|---:|---:|
| Training | 905 | 37.13% |
| Test | 227 | 32.16% |

Earlier encounters are used for model training and later encounters are reserved for evaluation.

This design preserves temporal ordering and provides a more realistic assessment of performance on later observations than a random row split.

The majority-class baseline accuracy in the test set is approximately **67.84%**.

---

## Evaluation Metrics

Model performance is evaluated using:

- accuracy;
- precision;
- recall;
- F1 score;
- ROC-AUC;
- PR-AUC; and
- confusion matrix.

Accuracy is not interpreted alone. Precision, recall, PR-AUC, and ROC-AUC are considered together when comparing the candidate models.

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 92.07% | 92.31% | 82.19% | 86.96% | 0.950 | 0.910 |
| Logistic Regression | 91.19% | 87.32% | 84.93% | 86.11% | 0.908 | 0.862 |

Using PR-AUC as the primary comparison metric and ROC-AUC as the secondary metric, **Random Forest is selected as the stronger candidate model**.

At a classification threshold of 0.50, the selected Random Forest produced:

- 149 true negatives;
- 60 true positives;
- 5 false positives; and
- 13 false negatives.

These results apply only to the held-out portion of this synthetic dataset and should not be interpreted as evidence of real-world clinical performance.

---

## Model Interpretation

Permutation importance on the held-out test data identifies several leading predictors, including:

1. `base_encounter_cost`
2. `total_claim_cost`
3. `days_since_previous_discharge`
4. `procedure_count`
5. `prior_readmissions`
6. `payer_coverage`

Feature importance reflects **predictive association rather than causality**.

The prominence of cost, procedure, and other completed-encounter variables also reinforces the discharge-time/post-encounter interpretation of the model.

---

## Analytical Risk Bands

Predicted probabilities from the selected model are grouped into:

| Probability | Risk Category |
|---|---|
| 0.00–0.30 | Low |
| >0.30–0.60 | Medium |
| >0.60–1.00 | High |

In the chronological test set:

| Risk Category | Encounters | Observed Readmission Rate |
|---|---:|---:|
| Low | 150 | 6.67% |
| Medium | 14 | 28.57% |
| High | 63 | 93.65% |

These bands are descriptive analytical groupings and are **not clinically validated thresholds**. The Medium group contains relatively few observations and should be interpreted cautiously.

---

## Known Limitations

1. The source data is synthetic and may not reproduce real clinical distributions.
2. The dataset is relatively small for predictive healthcare modeling.
3. Important clinical predictors such as laboratory results, medication history, detailed diagnoses, disease severity, and discharge disposition may not be available.
4. Readmissions occurring outside the encounters represented in the source data cannot be observed.
5. Several strong predictors are fully known only during or by the end of the index hospitalization.
6. Performance may vary across time periods and patient subgroups.
7. Demographic and payer variables require careful subgroup and fairness evaluation before any real-world predictive use.
8. The analytical probability bands are not established clinical thresholds.
9. Performance observed on this synthetic dataset should not be generalized to real hospital populations.

---

## Responsible Interpretation

A high predicted probability represents only the output of the fitted model within this synthetic analytical exercise.

It does not establish a patient's medical risk, provide a diagnosis, recommend an intervention, or justify a clinical or insurance decision.

Permutation importance and other observed associations should not be interpreted as causal relationships.

---

## Reproducibility

The workflow includes:

- a fixed random seed of 42 where applicable;
- preprocessing embedded in scikit-learn pipelines;
- chronological train/test validation;
- an explicit right-censoring rule;
- leakage-aware construction of prior-utilization variables; and
- reusable data-preparation and modeling logic in `src/`.

Additional implementation details are documented in `METHODOLOGY.md`.
