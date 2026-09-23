"""
app_streamlit.py
Credit Score Predictor — Streamlit App (direct model, no API)

Run:
    streamlit run app_streamlit.py
"""

import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pathlib import Path

# Allow Preprocessor class to be unpickled
sys.path.insert(0, str(Path(__file__).parent / "src" / "pipeline"))
from pipeline import Preprocessor  # noqa: F401

# PAGE CONFIG
st.set_page_config(
    page_title="Credit Score Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# PATHS & MODEL LOADING
BASE_DIR    = Path(__file__).parent
MODEL_PATH  = BASE_DIR / "models" / "best_model.pkl"
PREP_PATH   = BASE_DIR / "models" / "preprocessor.pkl"
DATA_PATH   = BASE_DIR / "data" / "C.csv"
NAME_PATH   = BASE_DIR / "models" / "best_model_name.txt"


@st.cache_resource
def load_models():
    clf  = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    prep = joblib.load(PREP_PATH)  if PREP_PATH.exists()  else None
    return clf, prep


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH) if DATA_PATH.exists() else None


clf_model, preprocessor = load_models()
df = load_data()
best_model_name = NAME_PATH.read_text().strip() if NAME_PATH.exists() else "—"

# SIDEBAR
st.sidebar.image("https://img.icons8.com/fluency/96/money.png", width=80)
st.sidebar.title("Credit Score (Michael Yeremia - 2802504876)")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigasi",
    ["Home", "Prediksi Individual", "Prediksi Batch (CSV)", "Analisis Dataset"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Status**")
st.sidebar.success(f"{best_model_name}" if clf_model else "Model belum tersedia")
st.sidebar.success("Preprocessor" if preprocessor else "Preprocessor belum tersedia")
if not clf_model or not preprocessor:
    st.sidebar.warning("Jalankan `python run_pipeline.py` terlebih dahulu.")


# HELPERS
def predict(input_dict: dict):
    df_in = pd.DataFrame([input_dict])
    X     = preprocessor.transform(df_in)
    idx   = clf_model.predict(X)[0]
    label = preprocessor.label_encoder.inverse_transform([idx])[0]
    proba_arr = clf_model.predict_proba(X)[0]
    classes   = preprocessor.label_encoder.classes_
    proba = {cls: round(float(p), 4) for cls, p in zip(classes, proba_arr)}
    return label, proba


def show_result(label: str, proba: dict):
    st.markdown("---")
    st.subheader("📊 Hasil Prediksi")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        color_map = {"Good": "success", "Standard": "warning", "Poor": "error"}
        icon_map  = {"Good": "✅", "Standard": "⚠️", "Poor": "❌"}
        icon  = icon_map.get(label, "")
        getattr(st, color_map.get(label, "info"))(
            f"{icon} **Credit Score: {label.upper()}**"
        )
        if label == "Good":
            st.info("Nasabah layak mendapatkan pinjaman dengan bunga kompetitif.")
        elif label == "Standard":
            st.info("Pertimbangkan batas pinjaman lebih rendah. Pantau pembayaran rutin.")
        else:
            st.info("Hindari pinjaman besar. Nasabah perlu memperbaiki riwayat pembayaran.")

        for cls in ["Good", "Standard", "Poor"]:
            p = proba.get(cls, 0)
            st.progress(float(p), text=f"{cls}: {p*100:.1f}%")

    with col_b:
        fig, ax = plt.subplots(figsize=(5, 3))
        classes = list(proba.keys())
        values  = list(proba.values())
        colors  = ["#2ECC71", "#F39C12", "#E74C3C"]
        bars = ax.barh(classes, values, color=colors, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, values):
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v*100:.1f}%", va="center", fontweight="bold")
        ax.set_xlim(0, 1.2)
        ax.set_title("Credit Score Probability", fontweight="bold")
        ax.set_xlabel("Probability")
        st.pyplot(fig)
        plt.close()


# PAGE: HOME
if page == "Home":
    st.title("Credit Score Prediction System")
    st.markdown("""
Selamat datang di **Credit Score Predictor** — aplikasi prediksi risiko kredit nasabah
berbasis Machine Learning untuk institusi keuangan.

---
### Fitur Aplikasi
| Halaman | Fungsi |
|---|---|
| Prediksi Individual | Input data satu nasabah → prediksi credit score |
| Prediksi Batch | Upload CSV → prediksi massal |
| Analisis Dataset | Visualisasi dan statistik dataset training |

### Kategori Credit Score
| Label | Deskripsi |
|---|---|
| ✅ **Good** | Risiko rendah — nasabah finansial sehat |
| ⚠️ **Standard** | Risiko sedang — perlu pemantauan |
| ❌ **Poor** | Risiko tinggi — hindari pinjaman besar |

### Model yang Digunakan
Dilatih menggunakan sklearn pipeline dengan OOP architecture dan MLflow tracking.
    """)

    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Nasabah", f"{len(df):,}")
        vc = df["Credit_Score"].value_counts()
        col2.metric("Good", f"{vc.get('Good', 0):,}")
        col3.metric("Standard", f"{vc.get('Standard', 0):,}")
        col4.metric("Poor", f"{vc.get('Poor', 0):,}")


# PAGE: PREDIKSI INDIVIDUAL
elif page == "Prediksi Individual":
    st.title("Prediksi Individual")
    st.markdown("Masukkan data finansial nasabah untuk mendapatkan prediksi credit score.")

    with st.form("prediction_form"):
        st.subheader("Data Pribadi & Pekerjaan")
        col1, col2, col3 = st.columns(3)
        with col1:
            age        = st.number_input("Age", 18, 100, 35)
            occupation = st.selectbox("Occupation", [
                "Scientist", "Teacher", "Engineer", "Entrepreneur", "Developer",
                "Lawyer", "Media_Manager", "Doctor", "Journalist", "Manager",
                "Accountant", "Musician", "Mechanic", "Writer", "Architect",
            ])
            annual_income = st.number_input("Annual Income ($)", 0.0, 500000.0, 60000.0, step=1000.0)
        with col2:
            monthly_salary = st.number_input("Monthly Inhand Salary ($)", 0.0, 50000.0, 4500.0, step=100.0)
            num_bank_acc   = st.number_input("Num Bank Accounts", 0, 20, 3)
            num_credit_card= st.number_input("Num Credit Cards", 0, 20, 4)
        with col3:
            interest_rate = st.number_input("Interest Rate (%)", 0, 100, 11)
            num_of_loan   = st.number_input("Num of Loans", 0, 20, 2)
            type_of_loan  = st.text_input("Type of Loan", value="Personal Loan, and Auto Loan")

        st.subheader("Riwayat Kredit")
        col4, col5, col6 = st.columns(3)
        with col4:
            delay_due    = st.number_input("Delay from Due Date (days)", 0, 100, 5)
            num_delayed  = st.number_input("Num of Delayed Payments", 0, 50, 3)
            changed_limit= st.number_input("Changed Credit Limit ($)", 0.0, 100.0, 5.0)
        with col5:
            num_inquiries = st.number_input("Num Credit Inquiries", 0, 30, 2)
            credit_mix    = st.selectbox("Credit Mix", ["Good", "Standard", "Bad"])
            outstanding   = st.number_input("Outstanding Debt ($)", 0.0, 10000.0, 800.0, step=50.0)
        with col6:
            utilization    = st.slider("Credit Utilization Ratio (%)", 0.0, 100.0, 28.5)
            history_yrs    = st.number_input("Credit History (Years)", 0, 50, 8)
            history_mths   = st.number_input("Credit History (Months)", 0, 11, 4)

        st.subheader("Perilaku Pembayaran")
        col7, col8 = st.columns(2)
        with col7:
            payment_min   = st.selectbox("Payment of Min Amount", ["Yes", "No", "NM"])
            total_emi     = st.number_input("Total EMI per Month ($)", 0.0, 5000.0, 150.0, step=10.0)
            amt_invested  = st.number_input("Amount Invested Monthly ($)", 0.0, 5000.0, 300.0, step=10.0)
        with col8:
            payment_beh   = st.selectbox("Payment Behaviour", [
                "High_spent_Small_value_payments",
                "Low_spent_Large_value_payments",
                "High_spent_Medium_value_payments",
                "Low_spent_Small_value_payments",
                "High_spent_Large_value_payments",
                "Low_spent_Medium_value_payments",
            ])
            monthly_bal   = st.number_input("Monthly Balance ($)", 0.0, 20000.0, 500.0, step=50.0)

        submitted = st.form_submit_button("🚀 Prediksi Sekarang", use_container_width=True)

    if submitted:
        if not clf_model or not preprocessor:
            st.error("Model belum tersedia. Jalankan `python run_pipeline.py` terlebih dahulu.")
        else:
            input_data = {
                "Age": str(age), "Occupation": occupation,
                "Annual_Income": str(annual_income),
                "Monthly_Inhand_Salary": monthly_salary,
                "Num_Bank_Accounts": num_bank_acc,
                "Num_Credit_Card": num_credit_card,
                "Interest_Rate": interest_rate,
                "Num_of_Loan": str(num_of_loan),
                "Type_of_Loan": type_of_loan,
                "Delay_from_due_date": delay_due,
                "Num_of_Delayed_Payment": str(num_delayed),
                "Changed_Credit_Limit": str(changed_limit),
                "Num_Credit_Inquiries": float(num_inquiries),
                "Credit_Mix": credit_mix,
                "Outstanding_Debt": str(outstanding),
                "Credit_Utilization_Ratio": utilization,
                "Credit_History_Age": f"{history_yrs} Years and {history_mths} Months",
                "Payment_of_Min_Amount": payment_min,
                "Total_EMI_per_month": total_emi,
                "Amount_invested_monthly": str(amt_invested),
                "Payment_Behaviour": payment_beh,
                "Monthly_Balance": monthly_bal,
            }
            try:
                label, proba = predict(input_data)
                show_result(label, proba)
            except Exception as e:
                st.error(f"Prediction error: {e}")


# PAGE: BATCH
elif page == "📊 Prediksi Batch (CSV)":
    st.title("📊 Prediksi Batch")
    st.markdown(
        "Upload file CSV dengan format yang sama seperti dataset training "
        "(kolom `Credit_Score` boleh ada atau tidak)."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.write(f"**{len(df_batch)} baris** terdeteksi. Preview:")
        st.dataframe(df_batch.head(), use_container_width=True)

        if st.button("Jalankan Prediksi Batch", use_container_width=True):
            if not clf_model or not preprocessor:
                st.error("Model belum tersedia.")
            else:
                with st.spinner("Memproses..."):
                    X_batch   = preprocessor.transform(df_batch)
                    preds_idx = clf_model.predict(X_batch)
                    labels    = preprocessor.label_encoder.inverse_transform(preds_idx)
                    probas    = clf_model.predict_proba(X_batch)
                    classes   = preprocessor.label_encoder.classes_

                    df_result = df_batch.copy()
                    df_result["predicted_credit_score"] = labels
                    for i, cls in enumerate(classes):
                        df_result[f"prob_{cls.lower()}"] = probas[:, i].round(4)

                vc = pd.Series(labels).value_counts()
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total", len(labels))
                col2.metric("Good",     vc.get("Good", 0))
                col3.metric("Standard", vc.get("Standard", 0))
                col4.metric("Poor",     vc.get("Poor", 0))

                st.dataframe(
                    df_result[["predicted_credit_score", "prob_good", "prob_standard", "prob_poor"]].head(20),
                    use_container_width=True
                )
                csv_out = df_result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Hasil Prediksi (CSV)",
                    data=csv_out, file_name="batch_predictions.csv", mime="text/csv",
                    use_container_width=True,
                )


# PAGE: ANALISIS DATASET
elif page == "Analisis Dataset":
    st.title("Analisis Dataset")

    if df is None:
        st.error("Dataset tidak ditemukan di `data/C.csv`.")
    else:
        tab1, tab2, tab3 = st.tabs(["Overview", "Distribusi", "Korelasi"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Statistik Deskriptif")
                num_df = df.select_dtypes(include=[np.number])
                st.dataframe(num_df.describe(), use_container_width=True)
            with col2:
                st.subheader("Distribusi Credit Score")
                counts = df["Credit_Score"].value_counts()
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.pie(
                    counts.values, labels=counts.index,
                    autopct="%1.1f%%",
                    colors=["#2ECC71", "#E74C3C", "#F39C12"],
                    startangle=90, wedgeprops={"edgecolor": "white"},
                )
                ax.set_title("Credit Score Distribution", fontweight="bold")
                st.pyplot(fig); plt.close()

        with tab2:
            feature = st.selectbox("Pilih Fitur Numerik", [
                "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card",
                "Interest_Rate", "Delay_from_due_date", "Credit_Utilization_Ratio",
                "Total_EMI_per_month", "Monthly_Balance",
            ])
            fig, ax = plt.subplots(figsize=(8, 4))
            color_map = {"Good": "#2ECC71", "Standard": "#F39C12", "Poor": "#E74C3C"}
            for label, color in color_map.items():
                subset = df[df["Credit_Score"] == label][feature]
                ax.hist(subset.dropna(), bins=25, alpha=0.55, color=color, label=label, density=True)
            ax.set_title(f"Distribusi {feature} by Credit Score", fontweight="bold")
            ax.legend(); ax.set_xlabel(feature)
            st.pyplot(fig); plt.close()

        with tab3:
            num_cols = [
                "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
                "Delay_from_due_date", "Credit_Utilization_Ratio",
                "Total_EMI_per_month", "Monthly_Inhand_Salary", "Monthly_Balance",
            ]
            avail = [c for c in num_cols if c in df.columns]
            corr  = df[avail].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(
                corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8},
            )
            ax.set_title("Correlation Heatmap", fontweight="bold")
            st.pyplot(fig); plt.close()
