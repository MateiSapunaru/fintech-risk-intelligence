from pathlib import Path
from datetime import datetime
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.engine import URL
from sqlalchemy.orm import declarative_base, sessionmaker


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_DB = os.getenv("POSTGRES_DB", "loan_scoring_db")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

DATABASE_URL = URL.create(
    drivername="postgresql+psycopg2",
    username=POSTGRES_USER,
    password=POSTGRES_PASSWORD,
    host=POSTGRES_HOST,
    port=int(POSTGRES_PORT),
    database=POSTGRES_DB,
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer, nullable=False)
    income = Column(Float, nullable=False)
    loan_amount = Column(Float, nullable=False)
    credit_score = Column(Float, nullable=False)
    years_experience = Column(Integer, nullable=False)
    gender = Column(String, nullable=False)
    education = Column(String, nullable=False)
    city = Column(String, nullable=False)
    employment_type = Column(String, nullable=False)

    prediction = Column(Integer, nullable=False)
    approval_probability = Column(Float, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


def init_db():
    Base.metadata.create_all(bind=engine)


def save_prediction(application_data: dict, prediction: int, approval_probability: float):
    session = SessionLocal()
    try:
        record = PredictionRecord(
            age=int(application_data["Age"]),
            income=float(application_data["Income"]),
            loan_amount=float(application_data["LoanAmount"]),
            credit_score=float(application_data["CreditScore"]),
            years_experience=int(application_data["YearsExperience"]),
            gender=str(application_data["Gender"]),
            education=str(application_data["Education"]),
            city=str(application_data["City"]),
            employment_type=str(application_data["EmploymentType"]),
            prediction=int(prediction),
            approval_probability=float(approval_probability),
        )

        session.add(record)
        session.commit()
        session.refresh(record)

        return record.id
    finally:
        session.close()


def get_recent_predictions(limit: int = 10):
    session = SessionLocal()
    try:
        records = (
            session.query(PredictionRecord)
            .order_by(PredictionRecord.created_at.desc())
            .limit(limit)
            .all()
        )

        return [
            {
                "id": r.id,
                "age": r.age,
                "income": r.income,
                "loan_amount": r.loan_amount,
                "credit_score": r.credit_score,
                "years_experience": r.years_experience,
                "gender": r.gender,
                "education": r.education,
                "city": r.city,
                "employment_type": r.employment_type,
                "prediction": r.prediction,
                "approval_probability": r.approval_probability,
                "created_at": r.created_at.isoformat(),
            }
            for r in records
        ]
    finally:
        session.close()