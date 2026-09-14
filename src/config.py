from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

RANDOM_STATE = 42
READMISSION_WINDOW_DAYS = 30

EXPECTED_FILES = {
    "patients": "patients.csv",
    "encounters": "encounters.csv",
    "procedures": "procedures.csv",
    "payers": "payers.csv",
    "organizations": "organizations.csv",
}
