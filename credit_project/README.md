# 💳 Credit Score Prediction

Aplikasi machine learning untuk memprediksi **credit score** nasabah institusi keuangan — **Good**, **Standard**, atau **Poor** — berdasarkan data finansial historis.

Proyek ini dibuat untuk mata kuliah **DTSC6012001 – Model Deployment**, School of Computer Science, BINUS University.

🔗 **Live Demo:** [Streamlit Cloud](#) *(update setelah deploy)*

---

## 📌 Fitur

- **EDA & Modelling** — eksplorasi data lengkap + 5 eksperimen model ML (`notebooks/`)
- **OOP Pipeline** — `Preprocessor`, `BaseTrainer`, `Evaluator` dengan tracking **MLflow** (`src/pipeline/pipeline.py`)
- **Cloud Pipeline** — training & deployment via **AWS SageMaker** (`src/pipeline/aws_pipeline.py`)
- **Web App** — antarmuka prediksi interaktif dengan **Streamlit** (`app_streamlit.py`)
- **REST API** — backend inferencing dengan **FastAPI** (`api/main.py`)

---

## 🗂️ Struktur Proyek

```
credit_project/
├── app_streamlit.py              # Streamlit app (direct model)
├── app_frontend.py               # Streamlit app (via FastAPI)
├── run_pipeline.py                # Entrypoint training
├── requirements.txt
├── api/
│   └── main.py                    # FastAPI backend
├── src/pipeline/
│   ├── pipeline.py                # OOP local pipeline + MLflow
│   └── aws_pipeline.py            # AWS SageMaker cloud pipeline
├── notebooks/
│   └── 01_eda_modelling.ipynb     # EDA + eksperimen model
├── models/
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   └── best_model_name.txt
├── data/
│   └── C.csv
└── local_vs_cloud_comparison.md   # Perbandingan local vs cloud
```

---

## 🚀 Cara Menjalankan

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Opsional) Training ulang model
```bash
python run_pipeline.py
```

### 3. Jalankan Streamlit app
```bash
streamlit run app_streamlit.py
```
Buka `http://localhost:8501`

### 4. (Opsional) Jalankan via FastAPI
```bash
uvicorn api.main:app --reload      # terminal 1 → http://localhost:8000/docs
streamlit run app_frontend.py      # terminal 2 → http://localhost:8501
```

---

## 📊 Hasil Model

| Model | Accuracy | F1 Weighted |
|---|---|---|
| **Random Forest ✓** | **0.72** | **0.72** |
| Gradient Boosting | 0.71 | 0.71 |
| Decision Tree | 0.68 | 0.68 |
| AdaBoost | 0.63 | 0.60 |
| Logistic Regression | 0.60 | 0.59 |

Model terbaik dipilih berdasarkan **F1 Weighted** untuk menangani class imbalance.

---

## 🧪 Test Cases

| Kategori | Karakteristik |
|---|---|
| 🟢 **Good** | Income tinggi, 0 keterlambatan, Credit Mix: Good |
| 🟡 **Standard** | Income menengah, keterlambatan sedang |
| 🔴 **Poor** | Income rendah, interest rate & keterlambatan tinggi |

---

## 🛠️ Tech Stack

`Python` `scikit-learn` `MLflow` `Streamlit` `FastAPI` `AWS SageMaker` `pandas`

---

## 👤 Author

DTSC6012001 – Model Deployment | BINUS University
