"""
app_frontend.py
Credit Score Predictor — Streamlit Frontend (via FastAPI)

Run:
    # Terminal 1 — start API backend
    uvicorn api.main:app --reload

    # Terminal 2 — start frontend
    streamlit run app_frontend.py
"""

import time
import json
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# PAGE CONFIG
st.set_page_config(
    page_title="Credit Score Predictor – Frontend",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# SIDEBAR: API CONFIG
st.sidebar.image("https://img.icons8.com/fluency/96/api.png", width=70)
st.sidebar.title("⚙️ API Configuration")
API_BASE = st.sidebar.text_input("FastAPI Base URL", value="http://localhost:8000")
st.sidebar.markdown("---")

# Health check
try:
    r = requests.get(f"{API_BASE}/health", timeout=3)
    if r.status_code == 200:
        health = r.json()
        st.sidebar.success("🟢 API Connected")
        st.sidebar.caption(f"Model: {health.get('classification_model', '—')}")
        st.sidebar.caption(f"Preprocessor: {health.get('preprocessor', '—')}")
    else:
        st.sidebar.error(f"🔴 API Error: {r.status_code}")
except Exception:
    st.sidebar.error("🔴 API tidak dapat dijangkau.\nPastikan FastAPI berjalan di URL di atas.")

page = st.sidebar.radio(
    "Navigasi",
    [
        "🔮 Skenario 1 – Good Credit",
        "📉 Skenario 2 – Poor Credit",
        "🛠️ Custom Input",
        "📊 Batch Prediction",
        "📋 API Response Inspector",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Arsitektur Decoupled**\n\n"
    "```\n[Streamlit Frontend]\n       ↓ HTTP POST\n[FastAPI Backend]\n       ↓ joblib.load\n[ML Model (.pkl)]\n```"
)

# PREDEFINED SCENARIOS
SCENARIO_GOOD = {
    "Age": "28", "Occupation": "Engineer",
    "Annual_Income": "90000.0", "Monthly_Inhand_Salary": 7000.0,
    "Num_Bank_Accounts": 2, "Num_Credit_Card": 3,
    "Interest_Rate": 7, "Num_of_Loan": "1",
    "Type_of_Loan": "Auto Loan",
    "Delay_from_due_date": 0, "Num_of_Delayed_Payment": "0",
    "Changed_Credit_Limit": "3.0", "Num_Credit_Inquiries": 1.0,
    "Credit_Mix": "Good", "Outstanding_Debt": "200.0",
    "Credit_Utilization_Ratio": 15.0,
    "Credit_History_Age": "6 Years and 0 Months",
    "Payment_of_Min_Amount": "No", "Total_EMI_per_month": 120.0,
    "Amount_invested_monthly": "500.0",
    "Payment_Behaviour": "Low_spent_Large_value_payments",
    "Monthly_Balance": 1200.0,
}

SCENARIO_POOR = {
    "Age": "46", "Occupation": "Teacher",
    "Annual_Income": "9786.86", "Monthly_Inhand_Salary": 951.57,
    "Num_Bank_Accounts": 9, "Num_Credit_Card": 10,
    "Interest_Rate": 34, "Num_of_Loan": "2",
    "Type_of_Loan": "Personal Loan, and Student Loan",
    "Delay_from_due_date": 58, "Num_of_Delayed_Payment": "14",
    "Changed_Credit_Limit": "15.53", "Num_Credit_Inquiries": 6.0,
    "Credit_Mix": "Standard", "Outstanding_Debt": "2149.9",
    "Credit_Utilization_Ratio": 23.45,
    "Credit_History_Age": "3 Years and 2 Months",
    "Payment_of_Min_Amount": "Yes", "Total_EMI_per_month": 11.56,
    "Amount_invested_monthly": "54.46",
    "Payment_Behaviour": "Low_spent_Medium_value_payments",
    "Monthly_Balance": 309.13,
}


# HELPERS
def call_api(endpoint: str, payload: dict):
    url = f"{API_BASE}{endpoint}"
    try:
        t0  = time.time()
        r   = requests.post(url, json=payload, timeout=10)
        elapsed = (time.time() - t0) * 1000
        if r.status_code == 200:
            return r.json(), elapsed
        st.error(f"❌ API Error {r.status_code}: {r.text}")
        return None, elapsed
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Tidak dapat terhubung ke {url}. Pastikan FastAPI berjalan.")
        return None, 0
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None, 0


def show_result(result: dict, elapsed: float):
    if result is None:
        return

    st.markdown(f"<small>⚡ Response time: {elapsed:.1f}ms</small>", unsafe_allow_html=True)

    label     = result.get("credit_score", "")
    p_good    = result.get("probability_good", 0)
    p_std     = result.get("probability_standard", 0)
    p_poor    = result.get("probability_poor", 0)

    col1, col2 = st.columns([1, 1])
    with col1:
        icon_map  = {"Good": "✅", "Standard": "⚠️", "Poor": "❌"}
        color_map = {"Good": "success", "Standard": "warning", "Poor": "error"}
        getattr(st, color_map.get(label, "info"))(
            f"{icon_map.get(label, '')} **{label.upper()}**"
        )

        st.progress(float(p_good),  text=f"Good: {p_good*100:.1f}%")
        st.progress(float(p_std),   text=f"Standard: {p_std*100:.1f}%")
        st.progress(float(p_poor),  text=f"Poor: {p_poor*100:.1f}%")

        rec = result.get("recommendation")
        if rec:
            st.caption(rec)

    with col2:
        fig, ax = plt.subplots(figsize=(5, 3))
        classes = ["Good", "Standard", "Poor"]
        values  = [p_good, p_std, p_poor]
        colors  = ["#2ECC71", "#F39C12", "#E74C3C"]
        bars = ax.barh(classes, values, color=colors, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, values):
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v*100:.1f}%", va="center", fontweight="bold")
        ax.set_xlim(0, 1.2)
        ax.set_title("Credit Score Probability", fontweight="bold")
        st.pyplot(fig)
        plt.close()

    with st.expander("🔍 Raw JSON Response"):
        st.json(result)


# PAGES

# SKENARIO 1: GOOD CREDIT
if page == "🔮 Skenario 1 – Good Credit":
    st.title("🔮 Skenario 1: Good Credit")
    st.markdown(
        "Skenario ini mensimulasikan nasabah dengan profil keuangan sehat: "
        "usia 28, Engineer, income $90.000, 0 keterlambatan, Credit Mix: Good."
    )

    with st.expander("📄 Lihat Data Input", expanded=True):
        st.json(SCENARIO_GOOD)

    if st.button("🎯 Predict Credit Score", use_container_width=True, type="primary"):
        result, elapsed = call_api("/predict/credit", SCENARIO_GOOD)
        show_result(result, elapsed)


# SKENARIO 2: POOR CREDIT
elif page == "📉 Skenario 2 – Poor Credit":
    st.title("📉 Skenario 2: Poor Credit")
    st.markdown(
        "Skenario ini mensimulasikan nasabah dengan profil keuangan bermasalah: "
        "income $9.786, interest rate 34%, 58 hari keterlambatan, 14 delayed payments."
    )

    with st.expander("📄 Lihat Data Input", expanded=True):
        st.json(SCENARIO_POOR)

    if st.button("🎯 Predict Credit Score", use_container_width=True, type="primary"):
        result, elapsed = call_api("/predict/credit", SCENARIO_POOR)
        show_result(result, elapsed)

    # Perbandingan
    st.markdown("---")
    st.subheader("📊 Perbandingan Skenario 1 vs Skenario 2")
    compare_df = pd.DataFrame({
        "Fitur": [
            "Annual Income ($)", "Interest Rate (%)", "Delay from Due (days)",
            "Num Delayed Payments", "Credit Mix", "Num Bank Accounts",
        ],
        "Good Credit": [90000, 7, 0, 0, "Good", 2],
        "Poor Credit":  [9786, 34, 58, 14, "Standard", 9],
    }).set_index("Fitur")
    st.dataframe(compare_df, use_container_width=True)


# CUSTOM INPUT
elif page == "🛠️ Custom Input":
    st.title("🛠️ Custom Input")
    st.markdown("Masukkan data nasabah secara manual untuk diprediksi via API.")

    with st.form("custom_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age        = st.number_input("Age", 18, 100, 35)
            occupation = st.selectbox("Occupation", [
                "Engineer", "Teacher", "Doctor", "Journalist", "Accountant",
                "Scientist", "Lawyer", "Manager", "Architect", "Entrepreneur",
            ])
            annual_inc = st.number_input("Annual Income ($)", 0.0, 500000.0, 60000.0)
            monthly_sal= st.number_input("Monthly Inhand Salary ($)", 0.0, 50000.0, 4500.0)
            num_bank   = st.number_input("Num Bank Accounts", 0, 20, 3)
        with col2:
            num_cc     = st.number_input("Num Credit Cards", 0, 20, 4)
            int_rate   = st.number_input("Interest Rate (%)", 0, 100, 11)
            num_loan   = st.number_input("Num of Loans", 0, 20, 2)
            delay_due  = st.number_input("Delay from Due (days)", 0, 100, 5)
            num_delayed= st.number_input("Num Delayed Payments", 0, 50, 3)
        with col3:
            credit_mix = st.selectbox("Credit Mix", ["Good", "Standard", "Bad"])
            outstanding= st.number_input("Outstanding Debt ($)", 0.0, 10000.0, 800.0)
            utilization= st.slider("Credit Utilization (%)", 0.0, 100.0, 28.5)
            hist_yrs   = st.number_input("Credit History (Years)", 0, 50, 8)
            hist_mths  = st.number_input("Credit History (Months)", 0, 11, 4)

        payment_min = st.selectbox("Payment of Min Amount", ["Yes", "No", "NM"])
        pay_beh     = st.selectbox("Payment Behaviour", [
            "High_spent_Medium_value_payments", "Low_spent_Large_value_payments",
            "High_spent_Small_value_payments", "Low_spent_Small_value_payments",
        ])
        monthly_bal = st.number_input("Monthly Balance ($)", 0.0, 20000.0, 500.0)

        submitted = st.form_submit_button("🚀 Kirim ke API", use_container_width=True)

    if submitted:
        payload = {
            "Age": str(age), "Occupation": occupation,
            "Annual_Income": str(annual_inc),
            "Monthly_Inhand_Salary": float(monthly_sal),
            "Num_Bank_Accounts": int(num_bank),
            "Num_Credit_Card": int(num_cc),
            "Interest_Rate": int(int_rate),
            "Num_of_Loan": str(int(num_loan)),
            "Type_of_Loan": "Personal Loan",
            "Delay_from_due_date": int(delay_due),
            "Num_of_Delayed_Payment": str(int(num_delayed)),
            "Changed_Credit_Limit": "5.0",
            "Num_Credit_Inquiries": 2.0,
            "Credit_Mix": credit_mix,
            "Outstanding_Debt": str(outstanding),
            "Credit_Utilization_Ratio": float(utilization),
            "Credit_History_Age": f"{int(hist_yrs)} Years and {int(hist_mths)} Months",
            "Payment_of_Min_Amount": payment_min,
            "Total_EMI_per_month": 150.0,
            "Amount_invested_monthly": "300.0",
            "Payment_Behaviour": pay_beh,
            "Monthly_Balance": float(monthly_bal),
        }
        result, elapsed = call_api("/predict/credit", payload)
        show_result(result, elapsed)


# BATCH PREDICTION
elif page == "📊 Batch Prediction":
    st.title("📊 Batch Prediction via API")
    st.markdown("Upload CSV → data dikirim ke endpoint `/predict/batch` → hasil ditampilkan.")

    uploaded = st.file_uploader("Upload CSV (maks. 100 baris)", type=["csv"])
    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.dataframe(df_batch.head(), use_container_width=True)

        if st.button("🚀 Kirim ke API /predict/batch", use_container_width=True, type="primary"):
            customers = df_batch.to_dict(orient="records")[:100]
            payload   = {"customers": customers}
            result, elapsed = call_api("/predict/batch", payload)
            if result:
                summary = result.get("summary", {})
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total", result["total"])
                col2.metric("Good",     summary.get("Good", 0))
                col3.metric("Standard", summary.get("Standard", 0))
                col4.metric("Poor",     summary.get("Poor", 0))

                df_res = pd.DataFrame(result["predictions"])
                st.dataframe(df_res, use_container_width=True)
                csv = df_res.to_csv(index=False).encode()
                st.download_button("⬇️ Download Hasil", csv, "batch_results.csv", "text/csv")


# API RESPONSE INSPECTOR
elif page == "📋 API Response Inspector":
    st.title("📋 API Response Inspector")
    st.markdown("Lihat raw JSON response dari endpoint. Berguna untuk debugging dan dokumentasi.")

    scenario = st.radio("Gunakan Skenario", ["Good Credit", "Poor Credit"], horizontal=True)
    payload  = SCENARIO_GOOD if scenario == "Good Credit" else SCENARIO_POOR

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📤 Request Payload")
        st.json(payload)
    with col2:
        if st.button("▶ Execute POST /predict/credit", type="primary", use_container_width=True):
            result, elapsed = call_api("/predict/credit", payload)
            st.subheader(f"📥 Response ({elapsed:.0f}ms)")
            if result:
                st.json(result)
                st.caption(f"POST {API_BASE}/predict/credit → 200 OK ({elapsed:.1f}ms)")
