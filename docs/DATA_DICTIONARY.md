# Data Dictionary

This document distinguishes between **source fields**, **intermediate engineered fields**, **final modeling fields**, and **model outputs** used in the 30-day hospital readmission analysis.

Column names are standardized to lowercase snake_case during data preparation.

---

## Source Tables 

### `patients.csv`

| Field | Role |
|---|---|
| `id` | Patient identifier |
| `birthdate` | Patient date of birth |
| `deathdate` | Date of death, when present |
| `gender` | Recorded gender |
| `race` | Recorded race |
| `ethnicity` | Recorded ethnicity |
| `marital` | Recorded marital status |

### `encounters.csv`

| Field | Role |
|---|---|
| `id` | Encounter identifier |
| `start` | Encounter start timestamp |
| `stop` | Encounter end timestamp |
| `patient` | Patient foreign key |
| `payer` | Payer foreign key |
| `encounterclass` | Encounter classification; inpatient encounters are used for target construction |
| `base_encounter_cost` | Base encounter cost |
| `total_claim_cost` | Total claim cost |
| `payer_coverage` | Amount covered by the payer |
| `description` | Encounter description, when present |
| `reasoncode` | Encounter reason code, when present |
| `reasondescription` | Encounter reason description, when present |

### `procedures.csv`

| Field | Role |
|---|---|
| `encounter` | Encounter foreign key |
| `patient` | Patient foreign key |
| `start` | Procedure start timestamp |
| `stop` | Procedure stop timestamp |
| `description` | Procedure description |
| `base_cost` | Procedure cost |
| `reasoncode` | Procedure reason code, when present |
| `reasondescription` | Procedure reason description, when present |

### `payers.csv`

| Field | Role |
|---|---|
| `id` | Payer identifier |
| `name` | Payer name |

### `organizations.csv`

The organizations table is included as one of the source tables and is reviewed during data understanding. Organization-level fields are not used as predictors in the final model.

---

## Intermediate Engineered Fields

These fields are created during data preparation or target construction but are **not final model predictors**.

| Field | Definition |
|---|---|
| `next_admission` | Start timestamp of the next inpatient admission for the same patient |
| `days_to_next_admission` | Number of days from the current discharge to the next inpatient admission |
| `previous_discharge` | Discharge timestamp of the patient's previous inpatient encounter |
| `admission_year` | Calendar year of the inpatient admission; used for descriptive and time-based analysis |

`next_admission` and `days_to_next_admission` use future information only for construction of the outcome variable. They are not supplied to the predictive models.

---

## Target Variable

| Field | Definition |
|---|---|
| `readmitted_30d` | Equals 1 when the next inpatient admission begins within 0–30 days after discharge; otherwise 0 |

Encounters without a complete 30-day follow-up period at the end of the observable data are excluded before modeling.

---

## Final Modeling Fields

### Patient Characteristics

| Field | Definition |
|---|---|
| `age_at_admission` | Patient age at the start of the current inpatient encounter |
| `gender` | Recorded gender |
| `race` | Recorded race |
| `ethnicity` | Recorded ethnicity |
| `marital` | Recorded marital status |

### Current Encounter Characteristics

| Field | Definition |
|---|---|
| `length_of_stay_days` | Number of days between encounter start and stop |
| `base_encounter_cost` | Base cost of the current inpatient encounter |
| `total_claim_cost` | Total claim cost for the current inpatient encounter |
| `payer_coverage` | Amount of the current encounter claim covered by the payer |
| `procedure_count` | Number of procedures linked to the current encounter |

Several current-encounter variables are fully known only during or by the end of hospitalization. The final model is therefore interpreted as a **discharge-time or post-encounter analytical model** rather than a strictly admission-time model.

### Prior Utilization Characteristics

| Field | Definition |
|---|---|
| `prior_inpatient_visits` | Number of observed inpatient encounters before the current encounter |
| `prior_readmissions` | Number of earlier encounters followed by a readmission within 30 days |
| `days_since_previous_discharge` | Number of days between the previous inpatient discharge and the current admission |
| `missing_prior_history` | Indicator identifying encounters without an observed previous inpatient discharge |

Historical utilization variables are constructed chronologically so that the current encounter's outcome is not included in its own prior-history features.

### Calendar Characteristics

| Field | Definition |
|---|---|
| `admission_month` | Calendar month of admission |
| `admission_dayofweek` | Day of week of admission, where Monday = 0 and Sunday = 6 |
| `admission_weekend` | Equals 1 for a Saturday or Sunday admission; otherwise 0 |

### Insurance Characteristic

| Field | Definition |
|---|---|
| `payer_name` | Descriptive payer name obtained by joining the encounter and payer tables |

---

## Model Outputs

These fields are produced after model training and are not model predictors.

| Field | Definition |
|---|---|
| `readmission_probability` | Predicted probability of 30-day inpatient readmission from the selected classifier |
| `risk_category` | Analytical Low, Medium, or High probability band derived from the predicted probability |

The analytical risk categories are:

| Probability | Risk Category |
|---|---|
| 0.00–0.30 | Low |
| >0.30–0.60 | Medium |
| >0.60–1.00 | High |

These probability bands are descriptive analytical groupings and are **not clinically validated thresholds**.

---

## Notes

- The source dataset is synthetic.
- Source fields with substantial missingness, including encounter and procedure reason fields, are not used as primary predictors in the final model.
- Future information is used only to construct the readmission outcome and is excluded from the predictor set.
- Demographic variables are retained for analytical comparison, but observed associations should not be interpreted as causal relationships.
