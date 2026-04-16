from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import shap

from src.database import init_db, save_prediction, get_recent_predictions


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_optuna.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]
explainer = shap.TreeExplainer(classifier)

init_db()

app = FastAPI(
    title="Fintech Risk Intelligence API",
    description="Explainable Loan Approval Scoring API",
    version="1.0.0",
)


class LoanApplication(BaseModel):
    Age: int = Field(..., example=35)
    Income: float = Field(..., example=55000)
    LoanAmount: float = Field(..., example=15000)
    CreditScore: float = Field(..., example=720)
    YearsExperience: int = Field(..., example=10)
    Gender: str = Field(..., example="Male")
    Education: str = Field(..., example="Masters")
    City: str = Field(..., example="New York")
    EmploymentType: str = Field(..., example="Salaried")


@app.get("/")
def root():
    return {"message": "Fintech Risk Intelligence API is running"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict(application: LoanApplication):
    try:
        application_dict = application.model_dump()
        input_df = pd.DataFrame([application_dict])

        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])

        record_id = save_prediction(
            application_data=application_dict,
            prediction=prediction,
            approval_probability=probability,
        )

        return {
            "decision": "Approved" if prediction == 1 else "Rejected",
            "prediction": prediction,
            "loan_approved": bool(prediction),
            "approval_probability": round(probability, 6),
            "record_id": record_id,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_proba")
def predict_proba(application: LoanApplication):
    try:
        input_df = pd.DataFrame([application.model_dump()])
        probabilities = model.predict_proba(input_df)[0]

        return {
            "probability_not_approved": round(float(probabilities[0]), 6),
            "probability_approved": round(float(probabilities[1]), 6),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain")
def explain(application: LoanApplication):
    try:
        input_df = pd.DataFrame([application.model_dump()])

        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])

        transformed = preprocessor.transform(input_df)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        feature_names = preprocessor.get_feature_names_out()
        transformed_df = pd.DataFrame(transformed, columns=feature_names)

        shap_values = explainer.shap_values(transformed_df)

        base_value = explainer.expected_value
        if hasattr(base_value, "__len__") and not isinstance(base_value, str):
            base_value = float(base_value[0])
        else:
            base_value = float(base_value)

        contributions = pd.DataFrame(
            {
                "feature": feature_names,
                "value": transformed_df.iloc[0].values,
                "shap_value": shap_values[0],
            }
        )

        contributions["abs_shap"] = contributions["shap_value"].abs()
        contributions = contributions.sort_values("abs_shap", ascending=False)

        top_contributors = []
        for _, row in contributions.head(5).iterrows():
            clean_feature = row["feature"].replace("num__", "").replace("cat__", "")
            clean_feature = clean_feature.replace("EmploymentType", "Employment Type")
            clean_feature = clean_feature.replace("CreditScore", "Credit Score")
            clean_feature = clean_feature.replace("LoanAmount", "Loan Amount")
            clean_feature = clean_feature.replace("YearsExperience", "Years of Experience")
            clean_feature = clean_feature.replace("_", " ")

            top_contributors.append(
                {
                    "feature": clean_feature.strip(),
                    "feature_value": (
                        float(row["value"]) if pd.notnull(row["value"]) else None
                    ),
                    "shap_value": round(float(row["shap_value"]), 6),
                    "impact": (
                        "increases approval"
                        if row["shap_value"] > 0
                        else "decreases approval"
                    ),
                }
            )

        return {
            "decision": "Approved" if prediction == 1 else "Rejected",
            "prediction": prediction,
            "approval_probability": round(probability, 6),
            "base_value": round(base_value, 6),
            "top_contributors": top_contributors,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/recent")
def recent_predictions(limit: int = 10):
    try:
        return {"predictions": get_recent_predictions(limit=limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))