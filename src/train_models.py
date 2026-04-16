import json
from pathlib import Path

import joblib
import pandas as pd

from src.data_loader import load_data
from src.preprocessing import split_data, build_preprocessor
from src.modeling import (
    build_logistic_regression_pipeline,
    build_RF_pipeline,
    build_xgboost_pipeline,
    evaluate_classification_model,
)
from src.tuning import tune_xgboost


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def save_artifact(obj, filename: str) -> Path:
    path = MODELS_DIR / filename
    joblib.dump(obj, path)
    print(f"Saved: {path}")
    return path


def train_and_save_all_models():
    df = load_data()

    X_train, X_test, y_train, y_test = split_data(df)
    preprocessor = build_preprocessor(X_train)

    print("\nRunning Optuna tuning for XGBoost...")
    study = tune_xgboost(
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train,
        n_trials=30,
        scoring="roc_auc",
    )

    print(f"Best XGBoost CV ROC-AUC: {study.best_value:.6f}")
    print("Best XGBoost params:")
    print(study.best_params)

    # Save Optuna metadata
    with open(MODELS_DIR / "xgboost_optuna_best_params.json", "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=4)

    with open(MODELS_DIR / "xgboost_optuna_best_score.txt", "w", encoding="utf-8") as f:
        f.write(str(study.best_value))

    models = {
        "logistic_regression.pkl": build_logistic_regression_pipeline(preprocessor),
        "random_forest.pkl": build_RF_pipeline(preprocessor),
        "xgboost_optuna.pkl": build_xgboost_pipeline(preprocessor, **study.best_params),
    }

    all_metrics = []

    for filename, model in models.items():
        print(f"\nTraining {filename} ...")
        model.fit(X_train, y_train)

        metrics, report, cm = evaluate_classification_model(
            model, X_train, X_test, y_train, y_test
        )

        save_artifact(model, filename)

        row = {"model_file": filename}
        row.update(metrics)
        all_metrics.append(row)

        print(f"Saved model: {filename}")
        print(pd.DataFrame([metrics]).T.rename(columns={0: "value"}))
        print("\nClassification report:")
        print(report)
        print("\nConfusion matrix:")
        print(cm)

    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv(MODELS_DIR / "model_metrics.csv", index=False)

    print("\nTraining complete.")
    print(results_df)

    return results_df, study


if __name__ == "__main__":
    train_and_save_all_models()