from pathlib import Path

import joblib
import pandas as pd

from config import PROCESSED_DATA_DIR, MODEL_DIR, RANDOM_STATE
from data_preparation import get_model_features
from modeling import chronological_split, build_models, evaluate_binary_classifier


def main():
    input_path = PROCESSED_DATA_DIR / "readmission_model_data.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} does not exist. Run notebooks 01-03 first."
        )

    df = pd.read_csv(input_path, parse_dates=["start", "stop"])
    features, numeric_features, categorical_features = get_model_features(df)

    modeling_df = df.dropna(subset=["readmitted_30d", "start"]).copy()
    train_df, test_df = chronological_split(modeling_df, "start", 0.80)

    X_train = train_df[features]
    y_train = train_df["readmitted_30d"].astype(int)
    X_test = test_df[features]
    y_test = test_df["readmitted_30d"].astype(int)

    models = build_models(numeric_features, categorical_features, RANDOM_STATE)
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_binary_classifier(model, X_test, y_test)
        results.append({"model": name, **metrics})

    results_df = pd.DataFrame(results).sort_values(
        ["pr_auc", "roc_auc"], ascending=False, na_position="last"
    )
    winner = results_df.iloc[0]["model"]
    best_model = models[winner]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "readmission_model.joblib")
    results_df.to_csv(PROCESSED_DATA_DIR / "model_comparison.csv", index=False)

    print(results_df.to_string(index=False))
    print(f"\nSaved best model: {winner}")


if __name__ == "__main__":
    main()
