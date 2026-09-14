from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with lowercase snake_case column names."""
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return out


def _find_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> str | None:
    """Return the first matching standardized column name."""
    cols = set(df.columns)
    for candidate in candidates:
        if candidate in cols:
            return candidate
    if required:
        raise KeyError(
            f"None of the expected columns {list(candidates)} were found. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def load_raw_tables(raw_data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all CSV files expected by the project."""
    expected = {
        "patients": "patients.csv",
        "encounters": "encounters.csv",
        "procedures": "procedures.csv",
        "payers": "payers.csv",
        "organizations": "organizations.csv",
    }
    missing = [name for name in expected.values() if not (raw_data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing raw data files: "
            + ", ".join(missing)
            + f". Place the Maven CSV files in: {raw_data_dir}"
        )

    tables = {}
    for name, filename in expected.items():
        tables[name] = standardize_columns(pd.read_csv(raw_data_dir / filename))
    return tables


def parse_dates(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Parse date/datetime fields that are present."""
    out = {k: v.copy() for k, v in tables.items()}

    for col in ["birthdate", "deathdate"]:
        if col in out["patients"].columns:
            out["patients"][col] = pd.to_datetime(out["patients"][col], errors="coerce")

    for table_name in ["encounters", "procedures"]:
        for col in ["start", "stop"]:
            if col in out[table_name].columns:
                out[table_name][col] = pd.to_datetime(
                    out[table_name][col], errors="coerce", utc=True
                )
    return out


def build_readmission_dataset(
    tables: Dict[str, pd.DataFrame],
    readmission_window_days: int = 30,
) -> pd.DataFrame:
    """
    Create an encounter-level modeling table for 30-day inpatient readmission.

    The outcome is 1 when the next inpatient encounter begins 0-30 days after
    the current inpatient encounter ends.

    To reduce right-censoring, encounters whose discharge occurs within the final
    readmission window of the dataset are excluded.
    """
    patients = tables["patients"].copy()
    encounters = tables["encounters"].copy()
    procedures = tables["procedures"].copy()
    payers = tables["payers"].copy()

    patient_fk = _find_column(encounters, ["patient", "patient_id"])
    encounter_id = _find_column(encounters, ["id", "encounter_id"])
    encounter_class = _find_column(encounters, ["encounterclass", "encounter_class"])
    start_col = _find_column(encounters, ["start"])
    stop_col = _find_column(encounters, ["stop"])

    encounters = encounters.dropna(subset=[patient_fk, start_col, stop_col]).copy()
    inpatient = encounters[
        encounters[encounter_class].astype(str).str.lower().eq("inpatient")
    ].copy()

    inpatient = inpatient.sort_values([patient_fk, start_col, encounter_id]).reset_index(drop=True)

    inpatient["length_of_stay_days"] = (
        inpatient[stop_col] - inpatient[start_col]
    ).dt.total_seconds() / 86400
    inpatient = inpatient[inpatient["length_of_stay_days"].ge(0)].copy()

    inpatient["next_admission"] = inpatient.groupby(patient_fk)[start_col].shift(-1)
    inpatient["days_to_next_admission"] = (
        inpatient["next_admission"] - inpatient[stop_col]
    ).dt.total_seconds() / 86400

    inpatient["readmitted_30d"] = (
        inpatient["days_to_next_admission"]
        .between(0, readmission_window_days, inclusive="both")
        .fillna(False)
        .astype(int)
    )

    dataset_end = inpatient[start_col].max()
    if pd.notna(dataset_end):
        cutoff = dataset_end - pd.Timedelta(days=readmission_window_days)
        inpatient = inpatient[inpatient[stop_col].le(cutoff)].copy()

    # Historical utilization features: created after chronological sorting.
    inpatient = inpatient.sort_values([patient_fk, start_col, encounter_id]).copy()
    inpatient["prior_inpatient_visits"] = inpatient.groupby(patient_fk).cumcount()
    inpatient["prior_readmissions"] = (
        inpatient.groupby(patient_fk)["readmitted_30d"]
        .transform(lambda s: s.shift(1).fillna(0).cumsum())
        .astype(float)
    )
    inpatient["previous_discharge"] = inpatient.groupby(patient_fk)[stop_col].shift(1)
    inpatient["days_since_previous_discharge"] = (
        inpatient[start_col] - inpatient["previous_discharge"]
    ).dt.total_seconds() / 86400

    # Patient demographics.
    patient_id = _find_column(patients, ["id", "patient", "patient_id"])
    rename_map = {patient_id: patient_fk}
    patients = patients.rename(columns=rename_map)

    demographic_cols = [
        c for c in [patient_fk, "birthdate", "gender", "race", "ethnicity", "marital"]
        if c in patients.columns
    ]
    model_df = inpatient.merge(
        patients[demographic_cols].drop_duplicates(subset=[patient_fk]),
        on=patient_fk,
        how="left",
    )

    # Age at admission.
    if "birthdate" in model_df.columns:
        admission_naive = model_df[start_col].dt.tz_localize(None)
        model_df["age"] = np.floor(
            (admission_naive - model_df["birthdate"]).dt.days / 365.25
        )
        model_df.loc[model_df["age"].lt(0) | model_df["age"].gt(120), "age"] = np.nan

    # Procedures per encounter.
    proc_encounter_fk = _find_column(
        procedures, ["encounter", "encounter_id"], required=False
    )
    if proc_encounter_fk:
        procedure_count = (
            procedures.groupby(proc_encounter_fk)
            .size()
            .rename("procedure_count")
            .reset_index()
        )
        model_df = model_df.merge(
            procedure_count,
            left_on=encounter_id,
            right_on=proc_encounter_fk,
            how="left",
        )
        model_df["procedure_count"] = model_df["procedure_count"].fillna(0)
    else:
        model_df["procedure_count"] = 0

    # Payer name.
    payer_fk = _find_column(model_df, ["payer", "payer_id"], required=False)
    payer_id = _find_column(payers, ["id", "payer", "payer_id"], required=False)
    payer_name = _find_column(payers, ["name", "payer_name"], required=False)

    if payer_fk and payer_id and payer_name:
        payer_lookup = payers[[payer_id, payer_name]].drop_duplicates().copy()
        payer_lookup = payer_lookup.rename(
            columns={payer_id: payer_fk, payer_name: "payer_name"}
        )
        model_df = model_df.merge(payer_lookup, on=payer_fk, how="left")
    elif "payer_name" not in model_df.columns:
        model_df["payer_name"] = "Unknown"

    # Financial features.
    claim_col = _find_column(
        model_df, ["total_claim_cost", "totalclaimcost"], required=False
    )
    coverage_col = _find_column(
        model_df, ["payer_coverage", "payercoverage"], required=False
    )
    base_cost_col = _find_column(
        model_df, ["base_encounter_cost", "baseencountercost"], required=False
    )

    if claim_col and claim_col != "total_claim_cost":
        model_df = model_df.rename(columns={claim_col: "total_claim_cost"})
    if coverage_col and coverage_col != "payer_coverage":
        model_df = model_df.rename(columns={coverage_col: "payer_coverage"})
    if base_cost_col and base_cost_col != "base_encounter_cost":
        model_df = model_df.rename(columns={base_cost_col: "base_encounter_cost"})

    if "total_claim_cost" in model_df.columns and "payer_coverage" in model_df.columns:
        model_df["patient_cost"] = (
            model_df["total_claim_cost"] - model_df["payer_coverage"]
        )
        model_df["insurance_coverage_pct"] = np.where(
            model_df["total_claim_cost"].gt(0),
            model_df["payer_coverage"] / model_df["total_claim_cost"],
            0.0,
        )
    else:
        model_df["patient_cost"] = np.nan
        model_df["insurance_coverage_pct"] = np.nan

    # Calendar fields.
    model_df["admission_year"] = model_df[start_col].dt.year
    model_df["admission_month"] = model_df[start_col].dt.month
    model_df["admission_dayofweek"] = model_df[start_col].dt.dayofweek
    model_df["weekend_admission"] = model_df["admission_dayofweek"].isin([5, 6]).astype(int)

    # Stable names used throughout the project.
    rename_stable = {}
    if patient_fk != "patient":
        rename_stable[patient_fk] = "patient"
    if encounter_id != "encounter_id":
        rename_stable[encounter_id] = "encounter_id"
    if start_col != "start":
        rename_stable[start_col] = "start"
    if stop_col != "stop":
        rename_stable[stop_col] = "stop"
    model_df = model_df.rename(columns=rename_stable)

    if "days_since_previous_discharge" in model_df.columns:
        model_df["days_since_previous_discharge_missing"] = (
            model_df["days_since_previous_discharge"].isna().astype(int)
        )

    return model_df.sort_values(["start", "patient", "encounter_id"]).reset_index(drop=True)


def get_model_features(df: pd.DataFrame) -> Tuple[list[str], list[str], list[str]]:
    """Return available feature sets: all, numeric, categorical."""
    numeric_candidates = [
        "age",
        "length_of_stay_days",
        "base_encounter_cost",
        "total_claim_cost",
        "payer_coverage",
        "patient_cost",
        "insurance_coverage_pct",
        "procedure_count",
        "prior_inpatient_visits",
        "prior_readmissions",
        "days_since_previous_discharge",
        "days_since_previous_discharge_missing",
        "admission_month",
        "admission_dayofweek",
        "weekend_admission",
    ]
    categorical_candidates = ["gender", "race", "ethnicity", "marital", "payer_name"]

    numeric = [c for c in numeric_candidates if c in df.columns]
    categorical = [c for c in categorical_candidates if c in df.columns]
    return numeric + categorical, numeric, categorical
