from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st

def _under_streamlit() -> bool:
    """False kalau dijalankan dengan `python app.py`. Di mode itu st.stop() tidak
    berfungsi sehingga error jadi menyesatkan."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return True


if not _under_streamlit():
    raise SystemExit(
        "\nApp ini harus dijalankan lewat Streamlit, bukan Python langsung.\n\n"
        "    streamlit run app.py\n"
    )


st.set_page_config(page_title="Retensi Pelanggan — TelcoConnect",
                   page_icon="📉", layout="wide")

NAVY, SAGE, BROWN = "#241268", "#6E9187", "#6F625B"
HIGH, MED, LOW = "#D9534F", "#E8973C", "#5B9E68"
BLUE, INK, MUTED = "#4A6FA5", "#1A1A1A", "#6B6B6B"

st.markdown(f"""
<style>
  .block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
  h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
  .tier-card {{ border-radius: 10px; padding: 18px 22px; color: #fff; }}
  .tier-name {{ font-size: 0.78rem; letter-spacing: .14em; text-transform: uppercase;
                opacity: .85; margin-bottom: 2px; }}
  .tier-prob {{ font-size: 2.6rem; font-weight: 700; line-height: 1.05; }}
  .tier-sub  {{ font-size: 0.86rem; opacity: .9; }}
  .note {{ background: #F4F5F6; border-left: 3px solid {SAGE};
           padding: 12px 16px; border-radius: 4px; font-size: 0.88rem; color: {INK}; }}
  .caveat {{ color: {MUTED}; font-size: 0.8rem; line-height: 1.5; }}
  div[data-testid="stMetricValue"] {{ font-size: 1.5rem; }}
</style>""", unsafe_allow_html=True)

# artifact
APP_DIR = Path(__file__).resolve().parent
ARTIFACT = APP_DIR / "churn_artifact.joblib"


def _die(*blocks):
    """Tampilkan pesan lalu benar-benar berhenti. SystemExit = jaring pengaman
    supaya script tidak pernah lanjut dengan variabel kosong."""
    st.error("\n\n".join(blocks))
    st.stop()
    raise SystemExit("\n\n".join(blocks))


@st.cache_resource(show_spinner="Memuat model…")
def _load_file(path: str, mtime: float):
    return joblib.load(path)


@st.cache_resource(show_spinner="Artifact tidak terpakai — melatih model dari CSV (±10 detik)…")
def _train_from_csv(path: str, mtime: float):
    from train_export import build_artifact
    return build_artifact(Path(path))


def _find_csv():
    for pat in ('WA_Fn-UseC_*Churn*.csv', '*Telco*Churn*.csv', '*telco*churn*.csv'):
        hits = sorted(APP_DIR.glob(pat))
        if hits:
            return hits[0]
    return None


def _get_artifact():
    """1) artifact  2) kalau gagal, latih dari CSV  3) kalau tidak ada juga, menyerah jelas."""
    load_err = None
    if ARTIFACT.exists():
        try:
            return _load_file(str(ARTIFACT), ARTIFACT.stat().st_mtime), None
        except Exception as e:
            load_err = e  # versi sklearn/numpy beda -> jatuh ke CSV

    csv = _find_csv()
    if csv is not None:
        try:
            A = _train_from_csv(str(csv), csv.stat().st_mtime)
            warn = (f"Artifact tidak dipakai (`{type(load_err).__name__}`) — model dilatih "
                    f"ulang dari `{csv.name}`. Hasilnya identik dengan notebook."
                    if load_err else
                    f"`churn_artifact.joblib` tidak ada — model dilatih dari `{csv.name}`. "
                    f"Hasilnya identik dengan notebook.")
            return A, warn
        except Exception as e:
            _die(f"**Gagal melatih model dari `{csv.name}`.**\n\n`{type(e).__name__}: {e}`",
                 "Pastikan CSV-nya dataset Telco Customer Churn yang utuh (7.043 baris, "
                 "21 kolom).")

    import sklearn
    if load_err is not None:
        _die(f"**Artifact gagal dimuat dan tidak ada CSV untuk melatih ulang.**\n\n"
             f"`{type(load_err).__name__}: {load_err}`",
             f"Versi scikit-learn kamu **{sklearn.__version__}**; artifact dibuat dengan "
             f"**1.8.0**.",
             "Pilih salah satu:\n\n```\npip install scikit-learn==1.8.0\n```\n\n"
             "atau taruh `WA_Fn-UseC_-Telco-Customer-Churn.csv` di folder ini — "
             "app akan melatih sendiri.")
    _die(f"**Model belum ada.** Tidak ada `churn_artifact.joblib` maupun CSV dataset "
         f"di `{APP_DIR}`.",
         "Taruh salah satunya di folder ini, atau jalankan:\n\n```\npython train_export.py\n```")


A, LOAD_WARNING = _get_artifact()

PRE, MODEL, CAL = A["preprocessor"], A["model"], A["calibrator"]
THR, HIGH_CUT = A["threshold"], A["tier_high_cut"]
M, COST = A["metrics"], A["cost"]
INPUT_COLS = A["input_columns"]
BINARY = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

# inti model
def _map_binary(X):
    X = X.copy()
    for c in BINARY:
        X[c] = X[c].map({'Yes': 1, 'No': 0})
    X['gender'] = X['gender'].map({'Male': 1, 'Female': 0})
    return X


def raw_score(df_raw):
    """Skor mentah model — dipakai untuk threshold, tier, dan urutan. Identik notebook."""
    return MODEL.predict_proba(PRE.transform(_map_binary(df_raw[INPUT_COLS])))[:, 1]


def calibrate(raw):
    """Skor mentah -> peluang churn sebenarnya. Monoton ketat, urutan tidak berubah."""
    raw = np.asarray(raw, dtype=float).reshape(-1, 1)
    return CAL.predict_proba(raw)[:, 1]


def tier_of(raw_p):
    if raw_p < THR:
        return "Low Risk", LOW
    if raw_p < HIGH_CUT:
        return "Medium Risk", MED
    return "High Risk", HIGH


PRETTY = {'tenure': 'Lama berlangganan', 'MonthlyCharges': 'Tagihan bulanan',
          'Contract': 'Jenis kontrak', 'InternetService': 'Layanan internet',
          'PaymentMethod': 'Metode pembayaran', 'OnlineSecurity': 'Online Security',
          'TechSupport': 'Tech Support', 'OnlineBackup': 'Online Backup',
          'DeviceProtection': 'Device Protection', 'MultipleLines': 'Multiple Lines',
          'StreamingTV': 'Streaming TV', 'StreamingMovies': 'Streaming Movies',
          'PaperlessBilling': 'Tagihan paperless', 'SeniorCitizen': 'Senior citizen',
          'Partner': 'Punya pasangan', 'Dependents': 'Punya tanggungan',
          'PhoneService': 'Layanan telepon', 'gender': 'Gender'}


def contributions(row_raw):
    """Uraikan log-odds jadi kontribusi per atribut asli, plus pengali odds."""
    z = PRE.transform(_map_binary(pd.DataFrame([row_raw])[INPUT_COLS]))[0]
    contrib = MODEL.coef_[0] * z
    agg = {}
    for name, c in zip(A["feature_names"], contrib):
        body = name.split("__", 1)[1]
        base = next((col for col in INPUT_COLS
                     if body == col or body.startswith(col + "_")), body)
        agg[base] = agg.get(base, 0.0) + c
    out = pd.DataFrame({"atribut": [PRETTY.get(k, k) for k in agg],
                        "logodds": list(agg.values())})
    out["odds"] = np.exp(out["logodds"])
    out = out[out["logodds"].abs() > 1e-9]
    return out.reindex(out["logodds"].abs().sort_values(ascending=False).index)


ACTIONS = {
    "High Risk": ("Hubungi bulan ini. Prioritas #1.",
                  "Tawarkan paket retensi paling agresif. Lihat daftar pemicu di bawah — "
                  "tawaran yang menyentuh pemicu teratas jauh lebih mungkin nyambung."),
    "Medium Risk": ("Masuk antrean, bukan prioritas.",
                    "Cukup nurturing ringan: cross-sell add-on atau ajakan pindah ke "
                    "auto-payment. Jangan habiskan diskon besar di sini."),
    "Low Risk": ("Jangan dihubungi.",
                 "Churn aktual kelompok ini hanya 3,7%. Promo ke sini membakar budget "
                 "tanpa mencegah apa pun."),
}


# sidebar
with st.sidebar:
    st.markdown("### Model yang dipakai")
    st.markdown(f"""
<div class="caveat">
<b>Logistic Regression</b> (L1, C=0.01, class_weight=balanced)<br>
Threshold operasional <b>{THR:.4f}</b><br>
F2-Score {M['f2']:.3f} &nbsp;|&nbsp; ROC-AUC {M['roc_auc']:.3f}<br>
Recall {M['recall']*100:.1f}% — {M['cm']['tp']} dari {M['cm']['tp']+M['cm']['fn']} churner tertangkap
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Tingkat risiko")
    st.markdown(f"""
<div class="caveat">
<span style="color:{LOW}">●</span> <b>Low</b> — {THR:.2f}<br>
<span style="color:{MED}">●</span> <b>Medium</b> — {THR:.2f} s/d {HIGH_CUT:.2f}<br>
<span style="color:{HIGH}">●</span> <b>High</b> — ≥ {HIGH_CUT:.2f}<br><br>
Tier ditentukan dari <b>skor mentah</b> model.
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
<div class="caveat">
Prediksi berlaku <b>1 bulan ke depan</b>. Model tidak melihat keluhan, gangguan
jaringan, atau NPS.
</div>""", unsafe_allow_html=True)


# header
st.markdown(f"<div style='color:{SAGE};font-size:.78rem;letter-spacing:.16em;"
            f"text-transform:uppercase;font-weight:600'>TelcoConnect · Retention</div>",
            unsafe_allow_html=True)
st.title("Siapa yang perlu dihubungi bulan ini?")
st.markdown(f"<p style='color:{MUTED};margin-top:-8px'>Memperkirakan peluang churn 1 bulan "
            f"ke depan, alasannya, dan tindakan yang sepadan dengan biayanya.</p>",
            unsafe_allow_html=True)

if LOAD_WARNING:
    st.info(LOAD_WARNING)

tab1, tab2, tab3 = st.tabs(["Satu pelanggan", "Daftar prioritas (CSV)", "Tentang model"])


# TAB 1
with tab1:
    st.markdown("#### Data pelanggan")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Demografi**")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Punya pasangan", ["No", "Yes"])
        depend = st.selectbox("Punya tanggungan", ["No", "Yes"])

    with c2:
        st.markdown("**Layanan**")
        phone = st.selectbox("Layanan telepon", ["Yes", "No"])
        if phone == "No":
            multi = "No phone service"
            st.caption("Multiple lines: *No phone service* (otomatis)")
        else:
            multi = st.selectbox("Multiple lines", ["No", "Yes"])
        internet = st.selectbox("Layanan internet", ["DSL", "Fiber optic", "No"])
        no_net = internet == "No"
        if no_net:
            st.caption("Add-on internet otomatis *No internet service*")
            sec = backup = devprot = tech = stv = smov = "No internet service"
        else:
            sec = st.selectbox("Online security", ["No", "Yes"])
            backup = st.selectbox("Online backup", ["No", "Yes"])
            devprot = st.selectbox("Device protection", ["No", "Yes"])

    with c3:
        st.markdown("**Kontrak & biaya**")
        if not no_net:
            tech = st.selectbox("Tech support", ["No", "Yes"])
            stv = st.selectbox("Streaming TV", ["No", "Yes"])
            smov = st.selectbox("Streaming movies", ["No", "Yes"])
        contract = st.selectbox("Kontrak", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Tagihan paperless", ["Yes", "No"])
        payment = st.selectbox("Metode pembayaran",
                               ["Electronic check", "Mailed check",
                                "Bank transfer (automatic)", "Credit card (automatic)"])

    c4, c5 = st.columns(2)
    tenure = c4.slider("Lama berlangganan (bulan)", 0, 72, 8)
    charges = c5.slider("Tagihan bulanan ($)", 18.0, 120.0, 79.0, 0.05)

    row = {"gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
           "Partner": partner, "Dependents": depend, "tenure": tenure,
           "PhoneService": phone, "MultipleLines": multi, "InternetService": internet,
           "OnlineSecurity": sec, "OnlineBackup": backup, "DeviceProtection": devprot,
           "TechSupport": tech, "StreamingTV": stv, "StreamingMovies": smov,
           "Contract": contract, "PaperlessBilling": paperless,
           "PaymentMethod": payment, "MonthlyCharges": charges}

    st.markdown("---")
    r = float(raw_score(pd.DataFrame([row]))[0])
    p = float(calibrate([r])[0])
    tname, tcol = tier_of(r)
    headline, detail = ACTIONS[tname]

    a1, a2 = st.columns([1, 2])
    with a1:
        st.markdown(f"""
<div class="tier-card" style="background:{tcol}">
  <div class="tier-name">{tname}</div>
  <div class="tier-prob">{p*100:.0f}%</div>
  <div class="tier-sub">perkiraan peluang churn 1 bulan ke depan</div>
</div>""", unsafe_allow_html=True)
        st.caption(f"Dari 100 pelanggan dengan profil seperti ini, sekitar **{p*100:.0f}** "
                   f"pergi bulan depan. Skor mentah model {r:.3f} "
                   f"({'≥' if r >= HIGH_CUT else '<'} {HIGH_CUT:.2f}) → tier {tname}.")
    with a2:
        st.markdown(f"##### {headline}")
        st.markdown(f"<p style='color:{MUTED};margin-top:-6px'>{detail}</p>",
                    unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        b1.metric("Revenue berisiko", f"${charges:,.2f}/bln")
        b2.metric("Biaya promo (20%)", f"${0.2*charges:,.2f}")
        b3.metric("Ekspektasi rugi bila diabaikan", f"${p*charges:,.2f}")
        if p > COST["fp_mult"]:
            st.caption(f"Peluang churn {p*100:.0f}% melebihi biaya promo yang setara "
                       f"{COST['fp_mult']*100:.0f}% tagihan → menghubungi lebih murah "
                       f"daripada mendiamkan.")
        else:
            st.caption(f"Peluang churn {p*100:.0f}% di bawah biaya promo yang setara "
                       f"{COST['fp_mult']*100:.0f}% tagihan → mendiamkan lebih murah.")

    # ---- kenapa
    st.markdown("#### Kenapa risikonya segini?")
    st.markdown("<div class='note'>Logistic Regression bisa dibuka isinya: tiap atribut "
                "mengalikan <i>odds</i> churn. <b>×lebih dari 1</b> menaikkan risiko (merah), "
                "<b>×kurang dari 1</b> menurunkan (biru). Ini untuk pelanggan <i>ini</i>, "
                "bukan rata-rata.</div>", unsafe_allow_html=True)
    st.write("")

    con = contributions(row)
    fig, ax = plt.subplots(figsize=(8.5, 0.44 * len(con) + 0.9), dpi=150)
    cols = [HIGH if v > 0 else BLUE for v in con["logodds"]]
    ypos = np.arange(len(con))[::-1]
    ax.barh(ypos, con["logodds"], color=cols, height=.62)
    ax.set_yticks(ypos, con["atribut"])
    span = max(con["logodds"].abs().max(), 0.05)
    for yp, lo, od in zip(ypos, con["logodds"], con["odds"]):
        off = span * 0.04
        ax.text(lo + (off if lo > 0 else -off), yp, f"×{od:.2f}",
                va="center", ha="left" if lo > 0 else "right", fontsize=9.5, color=INK)
    ax.axvline(0, color="#333", lw=.9)
    ax.set_xlim(-span * 1.42, span * 1.42)
    ax.set_xlabel("pengaruh ke odds churn")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="x", color="#EDEDED", lw=.7)
    ax.set_axisbelow(True)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(f"Model memakai {M['n_features_alive']} dari {M['n_features_total']} fitur — "
               f"L1 menolkan sisanya. Atribut yang tidak muncul memang tidak dipakai model "
               f"sama sekali.")

    # ---- pembanding
    st.markdown("#### Pelanggan serupa yang berbeda satu hal")
    st.markdown("<div class='note'>Ini <b>perbandingan, bukan janji</b>. Model belajar dari "
                "asosiasi, bukan sebab-akibat: angka di bawah menunjukkan risiko pelanggan "
                "yang profilnya sama persis tapi satu atributnya berbeda. Bukan berarti "
                "mengubah atribut itu <i>menyebabkan</i> risikonya turun sebesar itu — "
                "orang yang mengaktifkan Online Security memang cenderung tipe yang berbeda. "
                "Pakai ini untuk memilih tawaran mana yang paling relevan, bukan untuk "
                "menjanjikan hasil kampanye.</div>", unsafe_allow_html=True)
    st.write("")

    levers = []
    if not no_net and sec == "No":
        levers.append(("Punya Online Security", {"OnlineSecurity": "Yes"}, True))
    if not no_net and tech == "No":
        levers.append(("Punya Tech Support", {"TechSupport": "Yes"}, True))
    if not no_net and sec == "No" and tech == "No":
        levers.append(("Punya keduanya (Security + Tech Support)",
                       {"OnlineSecurity": "Yes", "TechSupport": "Yes"}, True))
    if payment == "Electronic check":
        levers.append(("Pakai auto-payment (bank transfer)",
                       {"PaymentMethod": "Bank transfer (automatic)"}, True))
    if contract == "Month-to-month":
        levers.append(("Kontrak 1 tahun", {"Contract": "One year"}, False))
    if internet == "Fiber optic":
        levers.append(("Pakai DSL", {"InternetService": "DSL"}, False))

    if not levers:
        st.info("Profil ini sudah memakai semua add-on & kontrak yang bisa ditawarkan — "
                "tidak ada pembanding tersisa. Kalau risikonya tetap tinggi, penyebabnya "
                "kemungkinan di luar data ini (keluhan, gangguan jaringan, harga).")
    else:
        rows = []
        for label, change, sellable in levers:
            r2 = float(raw_score(pd.DataFrame([{**row, **change}]))[0])
            p2 = float(calibrate([r2])[0])
            rows.append({"Profil pembanding": label,
                         "Peluang churn": f"{p2*100:.0f}%",
                         "Selisih": f"{(p2-p)*100:+.0f} pp",
                         "Tier": tier_of(r2)[0],
                         "Bisa ditawarkan?": "Ya" if sellable else "Bukan tawaran retensi",
                         "_d": p2 - p})
        st.dataframe(pd.DataFrame(rows).sort_values("_d").drop(columns="_d"),
                     use_container_width=True, hide_index=True)
        st.caption("“Kontrak 1 tahun” dan “Pakai DSL” ditandai bukan tawaran retensi: yang "
                   "pertama keputusan komersial berisiko (mengunci pelanggan yang sedang "
                   "ingin pergi), yang kedua keputusan produk/jaringan. Perlakukan sebagai "
                   "sinyal untuk tim terkait, bukan skrip telepon.")


# TAB 2
with tab2:
    st.markdown("#### Skor satu daftar pelanggan sekaligus")
    st.markdown("<div class='note'>Unggah CSV dengan kolom seperti dataset sumber. "
                "<code>customerID</code> opsional (dipakai sebagai label). "
                "<code>Churn</code> dan <code>TotalCharges</code> kalau ada akan "
                "diabaikan untuk prediksi.</div>", unsafe_allow_html=True)
    st.write("")

    up = st.file_uploader("Pilih file CSV", type=["csv"])
    if up is not None:
        raw_df = pd.read_csv(up)
        missing = [c for c in INPUT_COLS if c not in raw_df.columns]
        if missing:
            st.error("CSV kekurangan kolom berikut, jadi belum bisa diskor:\n\n"
                     + ", ".join(f"`{c}`" for c in missing))
        else:
            d = raw_df.copy()
            d["_raw"] = raw_score(d)
            d["_p"] = calibrate(d["_raw"].values)
            d["Tier"] = pd.cut(d["_raw"], [-np.inf, THR, HIGH_CUT, np.inf],
                               labels=["Low Risk", "Medium Risk", "High Risk"],
                               right=False)
            d["Peluang churn (%)"] = (d["_p"] * 100).round(0)
            d["Revenue berisiko ($)"] = (d["_p"] * d["MonthlyCharges"]).round(2)
            d["Biaya promo ($)"] = (COST["fp_mult"] * d["MonthlyCharges"]).round(2)

            n = len(d)
            contacted = d[d["_raw"] >= THR]
            budget_full = contacted["Biaya promo ($)"].sum()
            blanket = d["Biaya promo ($)"].sum()

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Pelanggan diskor", f"{n:,}")
            k2.metric("Di atas ambang", f"{len(contacted):,}", f"{len(contacted)/n*100:.0f}%")
            k3.metric("Biaya promo bila semua dihubungi", f"${budget_full:,.0f}")
            k4.metric("Revenue berisiko total", f"${d['Revenue berisiko ($)'].sum():,.0f}")

            st.markdown("##### Sebaran tier")
            dist = (d.groupby("Tier", observed=True)
                      .agg(Pelanggan=("_p", "size"),
                           **{"Rata-rata peluang churn (%)": ("Peluang churn (%)", "mean"),
                              "Revenue berisiko ($)": ("Revenue berisiko ($)", "sum")})
                      .reset_index())
            dist["Rata-rata peluang churn (%)"] = dist["Rata-rata peluang churn (%)"].round(0)
            dist["Revenue berisiko ($)"] = dist["Revenue berisiko ($)"].round(0)
            st.dataframe(dist, use_container_width=True, hide_index=True)

            # ---- perencana budget
            st.markdown("##### Budget kamu berapa bulan ini?")
            st.caption("Kalau budget tidak cukup untuk semua, app memilih urutan yang "
                       "melindungi revenue paling banyak per dolar promo.")
            cap = st.slider("Budget promo ($)", 0, int(np.ceil(budget_full)) or 1,
                            int(np.ceil(budget_full)), step=max(1, int(budget_full // 100)))

            plan = contacted.sort_values("Revenue berisiko ($)", ascending=False).copy()
            plan["_cum"] = plan["Biaya promo ($)"].cumsum()
            plan = plan[plan["_cum"] <= cap]

            p1, p2_, p3 = st.columns(3)
            p1.metric("Bisa dihubungi", f"{len(plan):,}",
                      f"dari {len(contacted):,} di atas ambang")
            p2_.metric("Terpakai", f"${plan['Biaya promo ($)'].sum():,.0f}",
                       f"{plan['Biaya promo ($)'].sum()/cap*100:.0f}% budget" if cap else "—")
            p3.metric("Revenue terlindungi", f"${plan['Revenue berisiko ($)'].sum():,.0f}",
                      f"{plan['Revenue berisiko ($)'].sum()/max(d['Revenue berisiko ($)'].sum(),1e-9)*100:.0f}% dari total berisiko")

            st.markdown("##### Urutan panggilan")
            st.caption("Diurutkan berdasarkan **revenue berisiko** (peluang churn × tagihan), "
                       "bukan peluang saja — pelanggan 60% dengan tagihan $110 lebih layak "
                       "didahulukan daripada 70% dengan tagihan $25.")
            show = (["customerID"] if "customerID" in d.columns else []) + [
                "Tier", "Peluang churn (%)", "MonthlyCharges", "tenure", "Contract",
                "InternetService", "OnlineSecurity", "TechSupport", "PaymentMethod",
                "Revenue berisiko ($)", "Biaya promo ($)"]
            out = plan[show]
            st.dataframe(out, use_container_width=True, hide_index=True, height=340)
            st.download_button("Unduh daftar panggilan (CSV)",
                               out.to_csv(index=False).encode(),
                               "daftar_panggilan.csv", "text/csv")

            if "Churn" in raw_df.columns:
                st.markdown("##### File ini punya kolom `Churn` — sekalian dievaluasi")
                yt = (raw_df["Churn"].map({"Yes": 1, "No": 0})
                      if raw_df["Churn"].dtype == object else raw_df["Churn"])
                yp = (d["_raw"] >= THR).astype(int)
                tp = int(((yp == 1) & (yt == 1)).sum()); fn = int(((yp == 0) & (yt == 1)).sum())
                fp = int(((yp == 1) & (yt == 0)).sum()); tn = int(((yp == 0) & (yt == 0)).sum())
                e1, e2, e3 = st.columns(3)
                e1.metric("Churner tertangkap", f"{tp} / {tp+fn}",
                          f"{tp/(tp+fn)*100:.1f}% recall" if tp + fn else "—")
                e2.metric("Terlewat (FN)", f"{fn}")
                e3.metric("Promo terbuang (FP)", f"{fp}")
                st.caption(f"TN {tn} · FP {fp} · FN {fn} · TP {tp}. Kalau ini data test yang "
                           f"sama dengan notebook, angkanya akan cocok. Kalau ini data baru "
                           f"dan recall jatuh jauh di bawah 90%, itu sinyal model drift → "
                           f"waktunya retrain.")


# TAB 3
with tab3:
    st.markdown("#### Performa di data test")
    cm = M["cm"]
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("F2-Score", f"{M['f2']:.3f}", f"{M['f2']-M['f2_default']:+.3f} vs threshold 0.5")
    g2.metric("ROC-AUC", f"{M['roc_auc']:.3f}")
    g3.metric("Recall", f"{M['recall']*100:.1f}%")
    g4.metric("Precision", f"{M['precision']*100:.1f}%")

    st.markdown(f"""
<div class="note">
Dari <b>{cm['tp']+cm['fn']} churner</b> di data test, model menangkap <b>{cm['tp']}</b> dan
melewatkan <b>{cm['fn']}</b>. Konsekuensinya <b>{cm['fp']}</b> pelanggan yang sebenarnya aman
ikut ditawari promo. Itu pertukaran yang disengaja: melewatkan churner (kehilangan seluruh
tagihan) jauh lebih mahal daripada memberi diskon 20% ke orang yang tidak akan pergi.
</div>""", unsafe_allow_html=True)

    # ---- kalibrasi
    st.markdown("#### Kenapa angka di app beda dengan skor model?")
    st.markdown(f"""
<div class="caveat">
<code>class_weight='balanced'</code> melatih model seolah churn 50:50, padahal aslinya
{A['baseline_churn_rate']*100:.1f}%. Skor mentahnya jadi menggelembung — rata-rata
<b>{M['mean_raw']*100:.1f}%</b> padahal churn aktual cuma <b>{M['base_rate_test']*100:.1f}%</b>.
Skor itu bagus untuk <i>mengurutkan</i> pelanggan, tapi tidak layak dikalikan dengan uang.<br><br>
App mengoreksinya dengan kalibrator Platt yang dilatih pada prediksi out-of-fold dari train set.
Brier score turun dari <b>{M['brier_raw']:.4f}</b> ke <b>{M['brier_cal']:.4f}</b>, rata-rata
prediksi jadi <b>{M['mean_cal']*100:.1f}%</b>. Karena transformasinya monoton ketat,
<b>urutan pelanggan, threshold, tier, dan confusion matrix persis sama dengan notebook</b> —
yang berubah hanya angka yang ditampilkan.
</div>""", unsafe_allow_html=True)
    st.write("")

    ct = pd.DataFrame(A["calibration_table"])
    ct_show = pd.DataFrame({
        "Skor mentah": ct["bin"].astype(str) + "%",
        "Pelanggan": ct["n"],
        "Kata skor mentah": (ct["mentah"] * 100).round(1),
        "Kata app (terkalibrasi)": (ct["kalibrasi"] * 100).round(1),
        "Churn aktual": (ct["aktual"] * 100).round(1),
    })
    st.dataframe(ct_show, use_container_width=True, hide_index=True)
    st.caption("Kolom terakhir adalah kebenarannya. Kolom “kata app” jauh lebih dekat ke "
               "sana daripada “kata skor mentah” — itulah gunanya kalibrasi.")

    # ---- threshold
    st.markdown(f"#### Kenapa threshold-nya {THR:.4f}, bukan 0,5?")
    st.caption("Threshold default 0,5 memperlakukan kedua jenis kesalahan sama beratnya. "
               "Padahal tidak. Kurva di bawah memakai biaya dari notebook: "
               "FN = 1,0 × tagihan · FP = 0,2 × tagihan.")

    cc = pd.DataFrame(A["cost_curve"])
    best_t = float(cc.loc[cc["cost"].idxmin(), "threshold"])
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=150)
    ax.plot(cc["threshold"], cc["cost"], color=NAVY, lw=2)
    ax.axvline(THR, color=HIGH, ls="--", lw=1.8, label=f"operasional (F2) = {THR:.2f}")
    ax.axvline(best_t, color=SAGE, ls=":", lw=1.8, label=f"biaya termurah = {best_t:.2f}")
    ax.axvline(0.5, color="#999", ls=":", lw=1.4, label="default = 0.50")
    ax.set_xlabel("threshold (skor mentah)"); ax.set_ylabel("total biaya FN+FP ($)")
    ax.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color="#EEE", lw=.7); ax.set_axisbelow(True)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown(f"""
<div class="caveat">
Jujurnya: threshold <b>{THR:.4f}</b> dipilih dengan memaksimalkan <b>F2-Score</b>, bukan
meminimalkan biaya. Titik biaya termurah ada di sekitar <b>{best_t:.2f}</b>. Keduanya sah —
F2 ditetapkan sejak awal proyek (Bab 1.5) dan tidak bergantung pada asumsi biaya yang masih
ilustratif. Kalau angka biaya riil sudah didapat dari Finance, threshold sebaiknya
dikalibrasi ulang ke titik biaya minimum.
</div>""", unsafe_allow_html=True)

    st.markdown("#### Yang tidak bisa dilakukan model ini")
    st.markdown(f"""
<div class="caveat">

- **Hanya 1 bulan ke depan.** Bukan prediksi jangka panjang, tidak menghitung kerugian
  seumur hidup pelanggan.
- **Asosiasi, bukan sebab-akibat.** Tabel pembanding di tab pertama menunjukkan risiko
  pelanggan *lain* yang atributnya berbeda — bukan efek dari mengubah atribut pelanggan ini.
- **Buta terhadap sinyal terpenting.** Tidak ada data keluhan, gangguan jaringan, atau NPS.
  Kalau kamu tahu pelanggan ini baru komplain, informasi kamu lebih akurat daripada model.
- **Data statis, bukan time-series.** Tidak ada validasi split berbasis waktu, jadi model
  drift tidak terdeteksi. Retrain berkala wajib.
- **Memakai {M['n_features_alive']} dari {M['n_features_total']} fitur.** L1 menolkan sisanya.
  Gender, senior citizen, dan streaming tidak berpengaruh sama sekali di model ini.
- **Angka biaya masih asumsi.** Diskon 20% dan retensi 100% efektif adalah ilustrasi,
  belum dikalibrasi ke data kampanye nyata.
- **Khusus telco.** Tidak bisa dipindah ke industri lain tanpa dilatih ulang.

</div>""", unsafe_allow_html=True)
