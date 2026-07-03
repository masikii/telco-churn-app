import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Konfigurasi halaman
st.set_page_config(
    page_title="Telco Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Styling ringan
st.markdown("""
<style>
    .main-header {font-size: 2.0rem; font-weight: 800; color: #1f2a44; margin-bottom: 0;}
    .sub-header {color: #5a6478; font-size: 1rem; margin-top: 0.2rem;}
    .risk-high {color: #C44E52; font-weight: 800;}
    .risk-med {color: #DD8452; font-weight: 800;}
    .risk-low {color: #55A868; font-weight: 800;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
</style>
""", unsafe_allow_html=True)

# Load model artifact (cached)
@st.cache_resource
def load_artifact(path="churn_model.joblib"):
    return joblib.load(path)

try:
    art = load_artifact()
except FileNotFoundError:
    st.error("File churn_model.joblib tidak ditemukan. Jalankan `python train_export.py` terlebih dahulu.")
    st.stop()

PIPELINE = art["pipeline"]
THRESHOLD = art["threshold"]
RISK_LOW = art["risk_low_bound"]
RISK_HIGH = art["risk_high_bound"]
FP_DISCOUNT = art["fp_discount"]
CAT_OPTIONS = art["cat_options"]
NUM_RANGES = art["num_ranges"]
FEATURE_ORDER = art["feature_order"]


def assign_risk_tier(p):
    if p < RISK_LOW:
        return "Low Risk"
    elif p < RISK_HIGH:
        return "Medium Risk"
    else:
        return "High Risk"


def risk_badge(tier):
    cls = {"Low Risk": "risk-low", "Medium Risk": "risk-med", "High Risk": "risk-high"}[tier]
    return f'<span class="{cls}">{tier}</span>'


def recommendation(tier):
    return {
        "High Risk": "Prioritas utama. Berikan penawaran retensi paling agresif (diskon/upgrade kontrak). ROI retensi tertinggi ada di segmen ini.",
        "Medium Risk": "Monitoring dan nudge ringan (email/SMS reminder, penawaran kecil). Hindari diskon besar karena keyakinan model masih moderat.",
        "Low Risk": "Tidak perlu aksi retensi proaktif. Cukup dipantau pada scoring bulan berikutnya.",
    }[tier]

# Header
st.markdown('<p class="main-header">Telco Customer Churn Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Memprediksi risiko churn pelanggan <b>1 bulan ke depan</b> dan memprioritaskan aksi retensi berdasarkan tingkat risiko.</p>', unsafe_allow_html=True)
st.divider()

# Sidebar — info model
with st.sidebar:
    st.header("Tentang Model")
    st.markdown(f"""
- **Algoritma:** Logistic Regression (F2-optimized)
- **Threshold operasional:** `{THRESHOLD:.2f}` (F2-optimal, sama dengan notebook)
- **F2-Score (test):** `{art['test_f2']:.3f}`
- **ROC-AUC (test):** `{art['test_auc']:.3f}`
""")
    st.divider()
    st.subheader("Tingkat Risiko")
    st.markdown(f"""
- **Low Risk** — probabilitas `< {RISK_LOW:.2f}`
- **Medium Risk** — `{RISK_LOW:.2f} sampai {RISK_HIGH:.2f}`
- **High Risk** — probabilitas `>= {RISK_HIGH:.2f}`
""")
    st.caption("Model dan threshold di-tuning (train-only). Alat bantu keputusan retensi, bukan pengganti kebijakan final.")

# Tabs
tab_single, tab_batch = st.tabs(["Prediksi Satu Pelanggan", "Prediksi Massal (CSV)"])

# TAB 1 — Single prediction
with tab_single:
    st.subheader("Masukkan Data Pelanggan")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demografi**")
        gender = st.selectbox("Gender", CAT_OPTIONS["gender"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", CAT_OPTIONS["Partner"])
        dependents = st.selectbox("Dependents", CAT_OPTIONS["Dependents"])

    with col2:
        st.markdown("**Layanan**")
        phone = st.selectbox("Phone Service", CAT_OPTIONS["PhoneService"])
        multiline = st.selectbox("Multiple Lines", CAT_OPTIONS["MultipleLines"])
        internet = st.selectbox("Internet Service", CAT_OPTIONS["InternetService"])
        online_sec = st.selectbox("Online Security", CAT_OPTIONS["OnlineSecurity"])
        online_bkp = st.selectbox("Online Backup", CAT_OPTIONS["OnlineBackup"])
        device = st.selectbox("Device Protection", CAT_OPTIONS["DeviceProtection"])

    with col3:
        st.markdown("**Layanan (lanjutan) dan Kontrak**")
        techsup = st.selectbox("Tech Support", CAT_OPTIONS["TechSupport"])
        stv = st.selectbox("Streaming TV", CAT_OPTIONS["StreamingTV"])
        smovies = st.selectbox("Streaming Movies", CAT_OPTIONS["StreamingMovies"])
        contract = st.selectbox("Contract", CAT_OPTIONS["Contract"])
        paperless = st.selectbox("Paperless Billing", CAT_OPTIONS["PaperlessBilling"])
        payment = st.selectbox("Payment Method", CAT_OPTIONS["PaymentMethod"])

    st.markdown("**Tenure dan Biaya**")
    c1, c2 = st.columns(2)
    with c1:
        t_min, t_max, t_med = NUM_RANGES["tenure"]
        tenure = st.slider("Tenure (bulan)", t_min, t_max, t_med)
    with c2:
        m_min, m_max, m_med = NUM_RANGES["MonthlyCharges"]
        monthly = st.slider("Monthly Charges ($)", float(m_min), float(m_max), float(m_med))

    if st.button("Prediksi Churn", type="primary", use_container_width=True):
        row = {
            'gender': gender, 'SeniorCitizen': 1 if senior == "Yes" else 0, 'Partner': partner,
            'Dependents': dependents, 'tenure': tenure, 'PhoneService': phone,
            'MultipleLines': multiline, 'InternetService': internet, 'OnlineSecurity': online_sec,
            'OnlineBackup': online_bkp, 'DeviceProtection': device, 'TechSupport': techsup,
            'StreamingTV': stv, 'StreamingMovies': smovies, 'Contract': contract,
            'PaperlessBilling': paperless, 'PaymentMethod': payment, 'MonthlyCharges': monthly,
        }
        Xrow = pd.DataFrame([row])[FEATURE_ORDER]
        proba = float(PIPELINE.predict_proba(Xrow)[:, 1][0])
        tier = assign_risk_tier(proba)
        pred_label = "AKAN CHURN" if proba >= THRESHOLD else "TIDAK CHURN"

        st.divider()
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.metric("Probabilitas Churn", f"{proba*100:.1f}%")
        with rc2:
            st.metric("Prediksi (threshold %.2f)" % THRESHOLD, pred_label)
        with rc3:
            st.markdown("**Tingkat Risiko**")
            st.markdown(f"### {risk_badge(tier)}", unsafe_allow_html=True)

        st.progress(min(proba, 1.0))

        if tier == "High Risk":
            st.error(f"**Rekomendasi:** {recommendation(tier)}")
        elif tier == "Medium Risk":
            st.warning(f"**Rekomendasi:** {recommendation(tier)}")
        else:
            st.success(f"**Rekomendasi:** {recommendation(tier)}")

        with st.expander("Konteks biaya (horizon 1 bulan)"):
            st.markdown(f"""
- Jika pelanggan ini **benar-benar churn** dan terlewat (False Negative) maka potensi revenue hilang **sekitar ${monthly:,.2f}** (1 bulan).
- Jika pelanggan ini **tidak churn** tetapi tetap diberi promo (False Positive) maka biaya promo sia-sia **sekitar ${FP_DISCOUNT*monthly:,.2f}** (diskon {FP_DISCOUNT:.0%}).
- Karena biaya False Negative jauh lebih besar dari False Positive, model dan threshold dirancang untuk menekan False Negative (menangkap sebanyak mungkin calon churner).
""")
            
# TAB 2 — Batch prediction
with tab_batch:
    st.subheader("Upload CSV Banyak Pelanggan")
    st.markdown(f"""
File CSV harus memuat kolom berikut (kolom customerID/TotalCharges boleh ada, akan diabaikan):
`{', '.join(FEATURE_ORDER)}`
""")

    template = pd.DataFrame(columns=FEATURE_ORDER)
    st.download_button("Download Template CSV Kosong",
                       template.to_csv(index=False).encode("utf-8"),
                       "template_churn.csv", "text/csv")

    uploaded = st.file_uploader("Pilih file CSV", type=["csv"])
    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Gagal membaca CSV: {e}")
            st.stop()

        missing = [c for c in FEATURE_ORDER if c not in df_in.columns]
        if missing:
            st.error(f"Kolom berikut tidak ada di CSV: {missing}")
            st.stop()

        work = df_in.copy()
        if work["SeniorCitizen"].dtype == object:
            work["SeniorCitizen"] = work["SeniorCitizen"].map({"Yes": 1, "No": 0}).fillna(work["SeniorCitizen"])
        Xb = work[FEATURE_ORDER].copy()

        with st.spinner("Menghitung prediksi..."):
            proba = PIPELINE.predict_proba(Xb)[:, 1]
        out = df_in.copy()
        out["Churn_Probability"] = np.round(proba, 4)
        out["Prediction"] = np.where(proba >= THRESHOLD, "Churn", "No Churn")
        out["Risk_Tier"] = [assign_risk_tier(p) for p in proba]

        st.divider()
        st.markdown("### Ringkasan Portofolio")
        summ = out["Risk_Tier"].value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"]).fillna(0).astype(int)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Pelanggan", len(out))
        s2.metric("High Risk", int(summ.get("High Risk", 0)))
        s3.metric("Medium Risk", int(summ.get("Medium Risk", 0)))
        s4.metric("Low Risk", int(summ.get("Low Risk", 0)))

        st.bar_chart(summ)

        st.markdown("### Detail (diurutkan dari risiko tertinggi)")
        out_sorted = out.sort_values("Churn_Probability", ascending=False).reset_index(drop=True)
        st.dataframe(out_sorted, use_container_width=True, height=420)

        st.download_button("Download Hasil Prediksi (CSV)",
                           out_sorted.to_csv(index=False).encode("utf-8"),
                           "hasil_prediksi_churn.csv", "text/csv", type="primary")

st.divider()
st.caption("Telco Churn Prediction - Logistic Regression + Risk Tier Segmentation - Horizon prediksi 1 bulan")
