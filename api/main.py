"""
api/main.py
Credit Score Prediction API — FastAPI backend

Run:
    uvicorn api.main:app --reload

Endpoints:
    GET  /                    → health check
    GET  /health              → model status
    POST /predict/credit      → predict credit score
    POST /predict/batch       → batch prediction
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional
import numpy as np
import pandas as pd
import joblib
import sys
from pathlib import Path

# Allow importing preprocessor class when unpickling
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "pipeline"))
from pipeline import Preprocessor  # noqa: F401 (needed for pickle)

# APP SETUP
app = FastAPI(
    title="Credit Score Prediction API",
    description="""
## 💳 Credit Score Prediction API

API ini memprediksi **Credit Score** nasabah berdasarkan data finansial.

### Endpoint Tersedia
- **`/predict/credit`** — Prediksi credit score satu nasabah
- **`/predict/batch`** — Prediksi massal (maks. 100 nasabah)

### Kategori Credit Score
| Label | Deskripsi |
|---|---|
| **Good** | Risiko kredit rendah — nasabah finansial sehat |
| **Standard** | Risiko kredit sedang — perlu perhatian |
| **Poor** | Risiko kredit tinggi — hindari pinjaman besar |

### 📌 Cara Uji di Swagger UI
1. Buka `/docs`
2. Pilih endpoint → klik **Try it out**
3. Masukkan JSON body → klik **Execute**
""",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# MODEL LOADING
BASE_DIR    = Path(__file__).parent.parent
MODEL_PATH  = BASE_DIR / "models" / "best_model.pkl"
PREP_PATH   = BASE_DIR / "models" / "preprocessor.pkl"
NAME_PATH   = BASE_DIR / "models" / "best_model_name.txt"


def _load(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


clf_model   = _load(MODEL_PATH)
preprocessor = _load(PREP_PATH)
best_model_name = NAME_PATH.read_text().strip() if NAME_PATH.exists() else "unknown"


# SCHEMAS (Pydantic)
class CustomerInput(BaseModel):
    Age: str = Field(..., example="35", description="Usia nasabah (18–100)")
    Occupation: str = Field(..., example="Engineer")
    Annual_Income: str = Field(..., example="75000.0")
    Monthly_Inhand_Salary: float = Field(..., example=5800.0)
    Num_Bank_Accounts: int = Field(..., ge=0, example=3)
    Num_Credit_Card: int = Field(..., ge=0, example=4)
    Interest_Rate: int = Field(..., ge=0, example=11)
    Num_of_Loan: str = Field(..., example="2")
    Type_of_Loan: str = Field(..., example="Personal Loan, and Auto Loan")
    Delay_from_due_date: int = Field(..., ge=0, example=5)
    Num_of_Delayed_Payment: str = Field(..., example="3")
    Changed_Credit_Limit: str = Field(..., example="5.0")
    Num_Credit_Inquiries: float = Field(..., ge=0, example=2.0)
    Credit_Mix: str = Field(..., example="Good", description="Good | Standard | Bad")
    Outstanding_Debt: str = Field(..., example="800.0")
    Credit_Utilization_Ratio: float = Field(..., ge=0, le=100, example=28.5)
    Credit_History_Age: str = Field(..., example="8 Years and 4 Months")
    Payment_of_Min_Amount: str = Field(..., example="No", description="Yes | No | NM")
    Total_EMI_per_month: float = Field(..., ge=0, example=150.0)
    Amount_invested_monthly: str = Field(..., example="300.0")
    Payment_Behaviour: str = Field(
        ..., example="High_spent_Medium_value_payments",
        description="Pola pembayaran nasabah"
    )
    Monthly_Balance: float = Field(..., example=500.0)

    @validator("Credit_Mix")
    def validate_credit_mix(cls, v):
        valid = {"Good", "Standard", "Bad", "_"}
        if v not in valid:
            raise ValueError(f"Credit_Mix harus salah satu dari: {valid}")
        return v

    @validator("Payment_of_Min_Amount")
    def validate_payment(cls, v):
        if v not in ("Yes", "No", "NM"):
            raise ValueError("Payment_of_Min_Amount harus Yes, No, atau NM")
        return v

    class Config:
        schema_extra = {
            "example": {
                "Age": "35", "Occupation": "Engineer",
                "Annual_Income": "75000.0", "Monthly_Inhand_Salary": 5800.0,
                "Num_Bank_Accounts": 3, "Num_Credit_Card": 4,
                "Interest_Rate": 11, "Num_of_Loan": "2",
                "Type_of_Loan": "Personal Loan, and Auto Loan",
                "Delay_from_due_date": 5, "Num_of_Delayed_Payment": "3",
                "Changed_Credit_Limit": "5.0", "Num_Credit_Inquiries": 2.0,
                "Credit_Mix": "Good", "Outstanding_Debt": "800.0",
                "Credit_Utilization_Ratio": 28.5,
                "Credit_History_Age": "8 Years and 4 Months",
                "Payment_of_Min_Amount": "No", "Total_EMI_per_month": 150.0,
                "Amount_invested_monthly": "300.0",
                "Payment_Behaviour": "High_spent_Medium_value_payments",
                "Monthly_Balance": 500.0,
            }
        }


class CreditScoreResponse(BaseModel):
    credit_score: str
    credit_score_label: str
    probability_good: float
    probability_standard: float
    probability_poor: float
    recommendation: str


class BatchInput(BaseModel):
    customers: list[CustomerInput]


# HELPERS
def to_dataframe(customer: CustomerInput) -> pd.DataFrame:
    return pd.DataFrame([customer.dict()])


def _check_models():
    if clf_model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model belum tersedia. Jalankan `python run_pipeline.py` terlebih dahulu.",
        )


def _make_recommendation(label: str, proba: dict) -> str:
    if label == "Good":
        return (
            f"Nasabah memiliki kredit BAIK (probabilitas {proba['Good']*100:.1f}%). "
            "Layak mendapatkan pinjaman dengan bunga kompetitif."
        )
    elif label == "Standard":
        return (
            f"Nasabah memiliki kredit STANDAR (probabilitas {proba['Standard']*100:.1f}%). "
            "Pertimbangkan batas pinjaman lebih rendah dan pantau pembayaran secara rutin."
        )
    else:
        return (
            f"Nasabah memiliki kredit BURUK (probabilitas {proba['Poor']*100:.1f}%). "
            "Hindari pinjaman besar. Nasabah perlu memperbaiki riwayat pembayaran."
        )


# ROUTES
@app.get("/", tags=["Health"])
def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Credit Score Prediction API is running",
        "model": best_model_name,
        "classification_model": "loaded" if clf_model else "not found",
        "preprocessor": "loaded" if preprocessor else "not found",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    """Cek status model."""
    return {
        "classification_model": best_model_name if clf_model else "not found – run run_pipeline.py",
        "preprocessor": "loaded" if preprocessor else "not found – run run_pipeline.py",
    }


@app.post("/predict/credit", response_model=CreditScoreResponse, tags=["Prediction"])
def predict_credit(customer: CustomerInput):
    """
    **Prediksi Credit Score** (Klasifikasi 3 Kelas)

    Menerima data finansial nasabah dan mengembalikan:
    - `credit_score`: label kelas (Good / Standard / Poor)
    - `probability_*`: probabilitas tiap kelas
    - `recommendation`: rekomendasi tindakan untuk institusi keuangan

    **Test Case 1 – Good (Nasabah finansial sehat):**
    Age=28, Occupation=Engineer, Annual_Income=90000, Interest_Rate=7,
    Delay_from_due_date=0, Num_of_Delayed_Payment=0, Credit_Mix=Good,
    Credit_History_Age=6 Years and 0 Months

    **Test Case 2 – Standard (Nasabah perlu pemantauan):**
    Age=35, Occupation=Teacher, Annual_Income=45000, Interest_Rate=14,
    Delay_from_due_date=10, Num_of_Delayed_Payment=5, Credit_Mix=Standard,
    Credit_History_Age=4 Years and 6 Months

    **Test Case 3 – Poor (Nasabah risiko tinggi):**
    Age=46, Occupation=Teacher, Annual_Income=9786, Interest_Rate=1663,
    Delay_from_due_date=58, Num_of_Delayed_Payment=14, Credit_Mix=Standard,
    Num_Bank_Accounts=9, Num_Credit_Card=10
    """
    _check_models()

    df_input = to_dataframe(customer)
    X        = preprocessor.transform(df_input)

    pred_idx  = int(clf_model.predict(X)[0])
    classes   = preprocessor.label_encoder.classes_
    label     = preprocessor.label_encoder.inverse_transform([pred_idx])[0]
    proba_arr = clf_model.predict_proba(X)[0]
    proba     = {cls: round(float(p), 4) for cls, p in zip(classes, proba_arr)}

    return CreditScoreResponse(
        credit_score=label,
        credit_score_label=label,
        probability_good=proba.get("Good", 0.0),
        probability_standard=proba.get("Standard", 0.0),
        probability_poor=proba.get("Poor", 0.0),
        recommendation=_make_recommendation(label, proba),
    )


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(batch: BatchInput):
    """
    **Prediksi Batch** – Kirim multiple nasabah dalam satu request.

    Maksimal 100 nasabah per request.
    Mengembalikan ringkasan dan detail prediksi setiap nasabah.
    """
    if len(batch.customers) > 100:
        raise HTTPException(status_code=400, detail="Maksimal 100 nasabah per request.")
    _check_models()

    classes = preprocessor.label_encoder.classes_
    results = []
    for customer in batch.customers:
        df_input  = to_dataframe(customer)
        X         = preprocessor.transform(df_input)
        pred_idx  = int(clf_model.predict(X)[0])
        label     = preprocessor.label_encoder.inverse_transform([pred_idx])[0]
        proba_arr = clf_model.predict_proba(X)[0]
        proba     = {cls: round(float(p), 4) for cls, p in zip(classes, proba_arr)}
        results.append({
            "credit_score": label,
            "probability_good": proba.get("Good", 0.0),
            "probability_standard": proba.get("Standard", 0.0),
            "probability_poor": proba.get("Poor", 0.0),
        })

    count = {cls: sum(1 for r in results if r["credit_score"] == cls) for cls in classes}
    return {
        "total": len(results),
        "summary": count,
        "distribution": {cls: round(count[cls] / len(results), 4) for cls in classes},
        "predictions": results,
    }
