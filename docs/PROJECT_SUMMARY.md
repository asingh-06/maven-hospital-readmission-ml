# Project Summary

## Predicting 30-Day Hospital Readmission Risk

This project uses Maven Analytics Hospital Patient Records to build a reproducible machine learning workflow for 30-day inpatient readmission analysis.

### Scope

The workflow:

- combines patient, encounter, procedure, and payer information;
- creates a 30-day inpatient readmission target from longitudinal encounters;
- excludes incomplete end-of-dataset follow-up periods;
- engineers utilization, demographic, financial, procedure, payer, and calendar features;
- compares Logistic Regression and Random Forest models;
- validates on later encounters using a chronological holdout;
- evaluates precision, recall, F1, ROC-AUC, and PR-AUC;
- produces encounter-level probability estimates and analytical risk bands.

### Key technical decisions

**Chronological validation**  
Earlier encounters are used for training and later encounters for testing.

**Leakage control**  
Historical counts are shifted so that current/future outcomes do not enter predictors.

**Right-censoring control**  
Encounters discharged during the final 30 observable days are excluded.

**Pipeline preprocessing**  
Imputation, scaling, and one-hot encoding are learned from training data and applied through scikit-learn pipelines.

### Output

The final analysis produces:

- a processed readmission modeling dataset;
- model comparison metrics;
- a saved fitted model;
- test-set readmission probabilities;
- permutation importance;
- charts for model evaluation.

### Data-use note

The source data is synthetic. Results are for analytical demonstration and should not be interpreted as clinically validated predictions.
