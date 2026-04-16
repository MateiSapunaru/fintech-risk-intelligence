from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import os


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_PATH = BASE_DIR / "models" / "model_metrics.csv"


def load_metrics() -> pd.DataFrame:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Metrics file not found at {METRICS_PATH}")

    df = pd.read_csv(METRICS_PATH)

    model_name_map = {
        "logistic_regression.pkl": "Logistic Regression",
        "random_forest.pkl": "Random Forest",
        "xgboost_optuna.pkl": "XGBoost (Optuna)",
    }

    df["Model"] = df["model_file"].map(model_name_map).fillna(df["model_file"])

    display_columns = {
        "test_accuracy": "Accuracy",
        "test_precision": "Precision",
        "test_recall": "Recall",
        "test_f1": "F1 Score",
        "test_roc_auc": "ROC AUC",
    }

    available_cols = ["Model"] + [col for col in display_columns if col in df.columns]
    df = df[available_cols].rename(columns=display_columns)

    metric_cols = [col for col in df.columns if col != "Model"]
    df[metric_cols] = df[metric_cols].round(3)

    return df


def load_recent_predictions(limit: int = 10) -> pd.DataFrame:
    response = requests.get(f"{API_URL}/predictions/recent", params={"limit": limit}, timeout=30)
    response.raise_for_status()

    payload = response.json()
    predictions = payload.get("predictions", [])

    if not predictions:
        return pd.DataFrame()

    df = pd.DataFrame(predictions)

    if "prediction" in df.columns:
        df["decision"] = df["prediction"].map({1: "Approved", 0: "Rejected"})

    if "approval_probability" in df.columns:
        df["approval_probability"] = df["approval_probability"].round(3)

    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    rename_map = {
        "id": "ID",
        "age": "Age",
        "income": "Income",
        "loan_amount": "Loan Amount",
        "credit_score": "Credit Score",
        "years_experience": "Years of Experience",
        "gender": "Gender",
        "education": "Education",
        "city": "City",
        "employment_type": "Employment Type",
        "decision": "Decision",
        "approval_probability": "Approval Probability",
        "created_at": "Created At",
    }

    keep_cols = [
        "id",
        "age",
        "income",
        "loan_amount",
        "credit_score",
        "years_experience",
        "gender",
        "education",
        "city",
        "employment_type",
        "decision",
        "approval_probability",
        "created_at",
    ]

    existing_cols = [col for col in keep_cols if col in df.columns]
    df = df[existing_cols].rename(columns=rename_map)

    return df


def prettify_feature_name(feature: str) -> str:
    feature = feature.replace("num__", "").replace("cat__", "")
    feature = feature.replace("EmploymentType", "Employment Type")
    feature = feature.replace("CreditScore", "Credit Score")
    feature = feature.replace("LoanAmount", "Loan Amount")
    feature = feature.replace("YearsExperience", "Years of Experience")
    feature = feature.replace("_", " ")
    return feature.strip()


def render_scoring_page() -> None:
    st.title("Fintech Risk Intelligence")
    st.subheader("Explainable Loan Approval Scoring")

    st.write(
        "Enter applicant information to generate a loan approval prediction "
        "and view the main explanatory factors."
    )

    with st.form("loan_application_form"):
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        income = st.number_input("Income", min_value=0.0, value=55000.0, step=1000.0)
        loan_amount = st.number_input("Loan Amount", min_value=0.0, value=15000.0, step=500.0)
        credit_score = st.number_input(
            "Credit Score", min_value=0.0, max_value=900.0, value=720.0, step=1.0
        )
        years_experience = st.number_input(
            "Years of Experience", min_value=0, max_value=60, value=10, step=1
        )

        gender = st.selectbox("Gender", ["Male", "Female"])
        education = st.selectbox(
            "Education", ["High School", "Bachelors", "Masters", "PhD"]
        )
        city = st.selectbox("City", ["New York", "Chicago", "Houston", "San Francisco"])
        employment_type = st.selectbox(
            "Employment Type", ["Salaried", "Self-Employed", "Unemployed"]
        )

        submitted = st.form_submit_button("Analyze Application")

    if not submitted:
        return

    payload = {
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "YearsExperience": years_experience,
        "Gender": gender,
        "Education": education,
        "City": city,
        "EmploymentType": employment_type,
    }

    try:
        predict_response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        explain_response = requests.post(f"{API_URL}/explain", json=payload, timeout=30)

        predict_response.raise_for_status()
        explain_response.raise_for_status()

        prediction_result = predict_response.json()
        explanation_result = explain_response.json()

        decision = prediction_result.get("decision", "Unknown")
        approval_probability = float(prediction_result.get("approval_probability", 0.0))
        record_id = prediction_result.get("record_id", "N/A")

        st.markdown("## Decision")

        if decision == "Approved":
            st.success(f"Decision: {decision}")
        elif decision == "Rejected":
            st.error(f"Decision: {decision}")
        else:
            st.warning(f"Decision: {decision}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Approval Probability", f"{approval_probability:.2%}")
        with col2:
            st.metric("Model Output", prediction_result.get("prediction", "N/A"))
        with col3:
            st.metric("Record ID", record_id)

        st.markdown("## Top Contributing Factors")

        contributors = explanation_result.get("top_contributors", [])
        if not contributors:
            st.info("No explanation factors were returned by the API.")
            return

        for item in contributors:
            feature = prettify_feature_name(str(item.get("feature", "Unknown feature")))
            impact = str(item.get("impact", "unknown impact"))
            shap_value = float(item.get("shap_value", 0.0))

            icon = "🟢" if "increase" in impact.lower() else "🔴"
            st.write(f"{icon} **{feature}** — {impact} (SHAP: {shap_value:.3f})")

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the API. Make sure FastAPI is running on "
            "http://127.0.0.1:8000"
        )
    except requests.exceptions.HTTPError as exc:
        st.error(f"API returned an error: {exc}")
        try:
            st.json(predict_response.json())
        except Exception:
            pass
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")


def render_model_comparison_page() -> None:
    st.title("Model Comparison")
    st.subheader("Performance on the test set")

    try:
        comparison_df = load_metrics()
    except Exception as exc:
        st.error(str(exc))
        return

    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    if "ROC AUC" in comparison_df.columns:
        best_model_row = comparison_df.sort_values("ROC AUC", ascending=False).iloc[0]
        best_model_name = best_model_row["Model"]
        st.success(
            f"Selected final model: **{best_model_name}**, based on the strongest "
            "overall ranking performance."
        )

    st.markdown("### Metric Notes")
    st.write(
        "- **Accuracy** measures overall correctness.\n"
        "- **Precision** shows how reliable positive approvals are.\n"
        "- **Recall** shows how many true approvals are captured.\n"
        "- **F1 Score** balances precision and recall.\n"
        "- **ROC AUC** measures ranking quality across thresholds."
    )


def render_recent_predictions_page() -> None:
    st.title("Recent Predictions")
    st.subheader("Latest applications stored in PostgreSQL")

    limit = st.slider("Number of recent predictions", min_value=5, max_value=50, value=10, step=5)

    try:
        recent_df = load_recent_predictions(limit=limit)
    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the API. Make sure FastAPI is running on "
            "http://127.0.0.1:8000"
        )
        return
    except requests.exceptions.HTTPError as exc:
        st.error(f"API returned an error: {exc}")
        return
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        return

    if recent_df.empty:
        st.info("No predictions have been stored yet.")
        return

    st.dataframe(recent_df, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="Fintech Risk Intelligence",
        page_icon="💳",
        layout="wide",
    )

    page = st.sidebar.radio(
        "Navigation",
        ["Loan Scoring", "Model Comparison", "Recent Predictions"],
    )

    if page == "Loan Scoring":
        render_scoring_page()
    elif page == "Model Comparison":
        render_model_comparison_page()
    else:
        render_recent_predictions_page()


if __name__ == "__main__":
    main()