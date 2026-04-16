# 💳 Fintech Risk Intelligence: Explainable Loan Approval Scoring

An end-to-end machine learning system for **credit risk assessment**, featuring **Explainable AI (XAI)**, a production-ready **API**, an interactive **dashboard**, and full **Dockerized deployment**.

---

## 📌 Project Overview

This project simulates a real-world **fintech credit scoring system**.

It predicts whether a loan application should be approved and provides **transparent explanations** for each decision using SHAP — a key requirement in regulated financial environments.

---

## 🧠 Key Features

* 🔍 **Loan Approval Prediction**

  * Binary classification (Approved / Rejected)
  * Probability-based scoring

* 📊 **Explainable AI (XAI)**

  * SHAP-based explanations
  * Top contributing factors per prediction

* ⚙️ **Model Optimization**

  * Logistic Regression (baseline)
  * Random Forest
  * **XGBoost optimized with Optuna (final model)**

* 🗄️ **Data Persistence**

  * PostgreSQL database
  * Stores all predictions for traceability

* 🌐 **API (FastAPI)**

  * `/predict`
  * `/explain`
  * `/predictions/recent`

* 🖥️ **Interactive Dashboard (Streamlit)**

  * User-friendly loan scoring interface
  * Model comparison page
  * Recent predictions viewer

* 🐳 **Dockerized Deployment**

  * Full system runs via Docker Compose
  * Reproducible and portable

---

## 🏗️ System Architecture

```text
User (Streamlit UI)
        ↓
   FastAPI Backend
        ↓
 ML Model (XGBoost + SHAP)
        ↓
 PostgreSQL Database
```

---

## 📸 Application Demo

### 🔹 Loan Scoring Interface

> User inputs applicant data and receives a decision + probability

📸 *(Add screenshot here — Streamlit main page with form + decision output)*

---

### 🔹 Explainability (SHAP)

> Each prediction is accompanied by interpretable feature contributions

📸 *(Add screenshot here — SHAP explanation / top contributing factors)*

---

### 🔹 Model Comparison

> Compare performance across different models

📸 *(Add screenshot here — model comparison table in Streamlit)*

---

### 🔹 Recent Predictions (Database)

> Stored predictions retrieved from PostgreSQL

📸 *(Add screenshot here — recent predictions page in Streamlit)*

---

### 🔹 API Documentation

> Interactive FastAPI docs for testing endpoints

📸 *(Add screenshot here — FastAPI /docs page)*

---

## 📊 Models & Performance

| Model               | Description                  |
| ------------------- | ---------------------------- |
| Logistic Regression | Baseline interpretable model |
| Random Forest       | Ensemble baseline            |
| XGBoost (Optuna)    | Final optimized model ✅      |

Model selection based on:

* ROC AUC
* F1 Score
* Generalization performance

---

## 🔍 Explainability (SHAP)

The model provides local explanations for each prediction:

* 🟢 Positive SHAP value → increases approval probability
* 🔴 Negative SHAP value → decreases approval probability

Example:

```text
🟢 Credit Score — increases approval
🔴 Loan Amount — decreases approval
```

This ensures **transparency and trust**, essential in financial systems.

---

## 🗄️ Database Design

Table: `predictions`

Stores:

* Applicant features
* Model prediction
* Approval probability
* Timestamp

Enables:

* Auditability
* Monitoring
* Future retraining pipelines

---

## 🚀 How to Run the Project

### 🔹 1. Clone the repository

```bash
git clone <your-repo-url>
cd fintech-risk-intelligence
```

---

### 🔹 2. Run with Docker

```bash
docker compose up --build
```

---

### 🔹 3. Access the application

* 📊 Streamlit UI → http://localhost:8501
* ⚙️ FastAPI Docs → http://localhost:8000/docs

---

## 🧪 API Endpoints

### 🔹 POST `/predict`

Returns:

* decision (Approved / Rejected)
* probability
* database record ID

---

### 🔹 POST `/explain`

Returns:

* prediction
* probability
* SHAP-based feature contributions

---

### 🔹 GET `/predictions/recent`

Returns:

* latest stored predictions from PostgreSQL

---

## 📁 Project Structure

```text
fintech-risk-intelligence/
│
├── api/                # FastAPI backend
├── app/                # Streamlit frontend
├── src/                # Core logic (DB, preprocessing, tuning)
├── models/             # Trained models + metrics
├── Dockerfile.api
├── Dockerfile.streamlit
├── docker-compose.yml
└── README.md
```

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* XGBoost
* Optuna
* SHAP
* FastAPI
* Streamlit
* PostgreSQL
* Docker

---

## 🎯 Key Learnings

This project demonstrates:

* End-to-end ML system design
* Model explainability in finance
* API development with FastAPI
* Interactive UI with Streamlit
* Database integration
* Containerized deployment

---

## 🚧 Future Improvements

* Model monitoring & drift detection
* Authentication & security layer
* CI/CD pipeline
* Cloud deployment (Azure / AWS)
* Real-world dataset integration

---

## 👨‍💻 Author

**Matei Săpunaru**
Machine Learning Engineer

---

## ⭐ Final Note

This project was built to simulate a **production-ready fintech ML system**, combining:

* machine learning
* explainability
* backend engineering
* frontend UX
* infrastructure

If you found this useful, feel free to ⭐ the repository.
