import os
import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import nbformat
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, balanced_accuracy_score
)

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Smart Soil Fertility - Klasifikasi Lahan",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background-color: #0f1715; color: #e8efe9; }
section[data-testid="stSidebar"] { background-color: #0a1411; }
section[data-testid="stSidebar"] * { color: #e8efe9 !important; }
.block-container { padding-top: 2rem; }
h1, h2, h3, h4 { color: #e8efe9; }
.metric-card {
    background: #16201c;
    border: 1px solid #1f2d27;
    border-radius: 12px;
    padding: 18px 20px;
}
.metric-card .label { color: #9bb0a4; font-size: 0.85rem; }
.metric-card .value { color: #e8efe9; font-size: 2rem; font-weight: 700; }
.hero {
    background: linear-gradient(135deg,#1f6b3a,#0f3d22);
    padding: 28px 32px; border-radius: 14px; color: #fff;
}
.hero h1 { color:#fff; margin:0 0 8px 0; }
.hero p  { color:#d6efd9; margin:0; }
.pill {
    display:inline-block; padding:4px 12px; border-radius:999px;
    font-size:0.85rem; margin-right:8px; margin-bottom:6px;
}
.pill-red    { background:#3a1a1a; color:#ff8b8b; border:1px solid #5a2424; }
.pill-yellow { background:#3a2f12; color:#ffcd6b; border:1px solid #5a4720; }
.pill-green  { background:#13331f; color:#7ee29a; border:1px solid #1f5c34; }
.feat-card {
    background:#16201c; border:1px solid #1f2d27;
    border-radius:10px; padding:14px 16px; height:100%;
}
.feat-card .name { font-weight:600; color:#e8efe9; }
.feat-card .unit { float:right; color:#9bb0a4; font-size:0.8rem; }
.feat-card .desc { color:#9bb0a4; font-size:0.85rem; margin-top:4px; }
div[role='radiogroup'] label {
    display: flex !important;
    align-items: center;
    padding: 0.6rem 0.75rem;
    border-radius: 8px;
    font-size: 0.95rem;
    cursor: pointer;
    transition: background .2s;
}
div[role='radiogroup'] label:hover {
    background: rgba(45,106,45,0.15);
}
div[role='radiogroup'] input[type='radio'] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "axes.facecolor": "#16201c",
    "figure.facecolor": "#16201c",
    "axes.edgecolor": "#2a3a33",
    "axes.labelcolor": "#cfd8d2",
    "xtick.color": "#cfd8d2",
    "ytick.color": "#cfd8d2",
    "text.color": "#e8efe9",
    "axes.grid": True,
    "grid.color": "#243029",
})

GREEN   = "#2fa05a"
ORANGE  = "#e8a13a"
RED     = "#d96a6a"
PALETTE = {0: RED, 1: ORANGE, 2: GREEN}
LABEL_NAME = {0: "Kurang Subur", 1: "Subur", 2: "Sangat Subur"}

# ============================================================
# DATA & MODEL — hanya didefinisikan SEKALI
# ============================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset1.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "RF.joblib")
    return joblib.load(model_path)

@st.cache_data
def evaluate_model(_model):
    """
    Evaluasi pakai split yang sama dengan training di Jupyter
    (random_state=42, test_size=0.2, stratify=y).
    Model TIDAK dilatih ulang — langsung predict.
    """
    df  = load_data()
    X   = df.drop("Output", axis=1)
    y   = df["Output"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred = _model.predict(X_test)

    return {
        "test_acc"    : accuracy_score(y_test, y_pred),
        "bal_acc"     : balanced_accuracy_score(y_test, y_pred),
        "f1"          : f1_score(y_test, y_pred, average="macro"),
        "report"      : classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "cm"          : confusion_matrix(y_test, y_pred, labels=[0, 1, 2]),
    }

# ============================================================
# SIDEBAR NAV
# ============================================================
with st.sidebar:
    st.markdown("""
        <div style='text-align:center; padding: 1rem 0 0.5rem;'>
            <div style='font-size:3rem'>🌱</div>
            <div style='font-weight:700; font-size:1.1rem; color:#4ade80;'>SoilSense</div>
            <div style='font-size:0.75rem; color:#888; margin-top:2px;'>Klasifikasi Lahan</div>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigasi",
        ["🏠 Beranda", "🔍 Eksplorasi Data", "📊 Klasifikasi Lahan", "📋 Tentang"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("v1.0.0 · SoilSense")

page = page.split(" ", 1)[1]

df    = load_data()
model = load_model()

# ============================================================
# PAGE: BERANDA
# ============================================================
def page_beranda():
    st.markdown("""
        <div class="hero">
          <h1>🌿 Smart Soil Fertility Predictor</h1>
          <p>Sistem cerdas untuk menentukan tingkat kesuburan tanah berdasarkan berbagai
           parameter kimia tanah secara cepat, praktis, dan mudah digunakan.</p>
        </div>
    """, unsafe_allow_html=True)
    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    counts = df["Output"].value_counts()
    cards = [
        ("Jumlah Sampel",  len(df)),
        ("Jumlah Fitur",   df.shape[1] - 1),
        ("Jumlah Kelas",   df["Output"].nunique()),
        ("Missing Values", int(df.isna().sum().sum())),
    ]
    for col, (label, val) in zip([c1, c2, c3, c4], cards):
        col.markdown(
            f"<div class='metric-card'><div class='value'>{val}</div>"
            f"<div class='label'>{label}</div></div>",
            unsafe_allow_html=True,
        )

    st.write("")
    st.subheader("Tentang Dataset")
    st.write(f"Dataset ini berisi **{len(df)} sampel** tanah dengan **{df.shape[1]-1} parameter kimia** "
             f"yang digunakan untuk mengklasifikasikan tingkat kesuburan lahan menjadi 3 kelas:")
    st.markdown(
        f"<span class='pill pill-red'>Kurang Subur ({counts.get(0,0)} sampel)</span>"
        f"<span class='pill pill-yellow'>Subur ({counts.get(1,0)} sampel)</span>"
        f"<span class='pill pill-green'>Sangat Subur ({counts.get(2,0)} sampel)</span>",
        unsafe_allow_html=True,
    )

    st.write("")
    st.subheader("Preview Data")

    col1, col2 = st.columns([3, 1])
    with col1:
        jumlah = st.slider("Jumlah baris", min_value=5, max_value=50, value=10, step=5)
    with col2:
        st.write("")
        acak = st.button("🔄 Acak", use_container_width=True)

    if "seed" not in st.session_state or acak:
        st.session_state.seed = random.randint(0, 9999)

    st.dataframe(df.sample(jumlah, random_state=st.session_state.seed), use_container_width=True)
    st.caption(f"Menampilkan {jumlah} dari {len(df)} baris")

    st.write("")
    st.subheader("Penjelasan Fitur")
    feats = [
        ("Nitrogen (N)",              "kg/ha", "Unsur makro utama untuk pertumbuhan vegetatif tanaman"),
        ("Phosphorus (P)",            "kg/ha", "Penting untuk perkembangan akar dan pembungaan"),
        ("Kalium (K)",                "kg/ha", "Meningkatkan ketahanan tanaman terhadap penyakit"),
        ("pH Tanah",                  "-",     "Tingkat keasaman/kebasaan tanah"),
        ("Electrical Conductivity",   "dS/m",  "Konduktivitas listrik tanah"),
        ("Organic Carbon (OC)",       "%",     "Kandungan karbon organik dalam tanah"),
        ("Sulfur (S)",                "ppm",   "Unsur penting untuk sintesis protein"),
        ("Zinc (Zn)",                 "ppm",   "Mikronutrien esensial untuk enzim tanaman"),
        ("Iron (Fe)",                 "ppm",   "Diperlukan untuk pembentukan klorofil"),
        ("Copper (Cu)",               "ppm",   "Berperan dalam fotosintesis dan respirasi"),
        ("Manganese (Mn)",            "ppm",   "Penting untuk metabolisme nitrogen"),
        ("Boron (B)",                 "ppm",   "Diperlukan untuk pembentukan dinding sel"),
    ]
    cols = st.columns(3)
    for i, (n, u, d) in enumerate(feats):
        with cols[i % 3]:
            st.markdown(
                f"<div class='feat-card'><span class='unit'>{u}</span>"
                f"<div class='name'>{n}</div><div class='desc'>{d}</div></div>",
                unsafe_allow_html=True,
            )
            st.write("")

# ============================================================
# PAGE: EKSPLORASI DATA
# ============================================================
def page_eksplorasi():
    st.title("Eksplorasi Data")
    st.caption("Visualisasi dan analisis karakteristik dataset kesuburan tanah")

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("Total Sampel",  len(df)),
        ("Fitur",         f"{df.shape[1]-1}"),
        ("Kelas Output",  df["Output"].nunique()),
        ("Missing Values",int(df.isna().sum().sum())),
    ]
    for col, (l, v) in zip([c1, c2, c3, c4], metrics):
        col.markdown(
            f"<div class='metric-card'><div class='value'>{v}</div><div class='label'>{l}</div></div>",
            unsafe_allow_html=True,
        )

    st.write("")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribusi Output")
        counts = df["Output"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = [PALETTE[c] for c in counts.index]
        ax.pie(counts.values, labels=[LABEL_NAME[c] for c in counts.index],
               colors=colors, autopct="%1.0f%%", wedgeprops=dict(width=0.4),
               textprops={"color": "#e8efe9"})
        st.pyplot(fig)

    with col2:
        st.subheader("Rata-rata NPK per Kelas")
        avg = df.groupby("Output")[["N", "P", "K"]].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        avg.index = [LABEL_NAME[i] for i in avg.index]
        avg.plot(kind="bar", ax=ax, color=["#2fa05a", "#e8a13a", "#5aa9d9"])
        ax.set_xlabel(""); ax.set_ylabel("")
        plt.xticks(rotation=0)
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Scatter: N vs K")
        fig, ax = plt.subplots(figsize=(6, 4))
        for cls, color in PALETTE.items():
            sub = df[df["Output"] == cls]
            ax.scatter(sub["N"], sub["K"], color=color, alpha=0.7, label=LABEL_NAME[cls])
        ax.set_xlabel("Nitrogen (N)"); ax.set_ylabel("Potassium (K)")
        ax.legend()
        st.pyplot(fig)

    with col4:
        st.subheader("Distribusi pH")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["pH"], bins=15, color=GREEN, edgecolor="#0f1715")
        ax.set_xlabel("pH"); ax.set_ylabel("Frekuensi")
        st.pyplot(fig)

    st.subheader("Distribusi Organic Carbon (OC)")
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.kdeplot(df["OC"], ax=ax, color=GREEN, fill=True)
    st.pyplot(fig)

    st.subheader("Distribusi Sulfur (S)")
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.kdeplot(df["S"], ax=ax, color=GREEN, fill=True)
    st.pyplot(fig)

    st.subheader("Heatmap Korelasi (Semua Fitur)")
    fig, ax = plt.subplots(figsize=(12, 8))
    corr = df.corr()
    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                    ha="center", va="center", color="black", fontsize=9)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticklabels(corr.columns)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Korelasi Antar Fitur", fontsize=14, weight="bold", pad=15)
    st.pyplot(fig)

    st.title("📓 Notebook Cell Viewer")
    nb_path = os.path.join(os.path.dirname(__file__), "KLASIFIKASI_LAHAN.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    for i, cell in enumerate(nb.cells):
        st.markdown(f"**Cell [{i+1}] — `{cell.cell_type}`**")
        if cell.cell_type == "markdown":
            st.markdown(cell.source)
        elif cell.cell_type == "code":
            st.code(cell.source, language="python")
            for output in cell.get("outputs", []):
                if output.output_type == "stream":
                    st.text(output.text)
                elif output.output_type in ("display_data", "execute_result"):
                    if "text/plain" in output.data:
                        st.text(output.data["text/plain"])
        st.divider()
# ============================================================
# PAGE: KLASIFIKASI LAHAN
# ============================================================
def page_klasifikasi():
    st.title("Klasifikasi Lahan")
    st.caption("Masukkan parameter tanah untuk memprediksi tingkat kesuburan")

    features = df.drop("Output", axis=1).columns.tolist()
    ranges   = {f: (float(df[f].min()), float(df[f].max())) for f in features}

    st.subheader("Input Parameter Tanah")

    # Counter untuk force re-render widget dengan key baru
    if "reset_counter" not in st.session_state:
        st.session_state.reset_counter = 0

    counter = st.session_state.reset_counter

    cols   = st.columns(4)
    inputs = {}
    for i, f in enumerate(features):
        lo, hi = ranges[f]
        with cols[i % 4]:
            inputs[f] = st.number_input(
                f"{f} ({lo:.2f} - {hi:.2f})",
                value=None,
                step=0.1,
                format="%.2f",
                placeholder="...",
                key=f"input_{f}_{counter}"   # key berubah tiap reset = widget baru
            )

    st.write("")
    b1, b2  = st.columns(2)
    do_pred = b1.button("🔍 Prediksi Kesuburan", use_container_width=True, type="primary")
    if b2.button("Reset", use_container_width=True):
        st.session_state.reset_counter += 1
        st.rerun()

    if do_pred:
        empty = [f for f in features if inputs[f] is None]
        if empty:
            st.warning("Harap isi semua input terlebih dahulu.")
        else:
            x     = pd.DataFrame([[inputs[f] for f in features]], columns=features)
            pred  = int(model.predict(x)[0])
            proba = model.predict_proba(x)[0]
            conf  = proba[list(model.classes_).index(pred)] * 100

            color_pill = {0: "pill-red", 1: "pill-yellow", 2: "pill-green"}
            rec = {
                0: "- Tanah memiliki kandungan unsur hara yang rendah sehingga pertumbuhan tanaman tidak optimal.<br>"
                   "- Kondisi tanah kurang mendukung karena nutrisi di bawah standar.<br>"
                   "- Tanah membutuhkan perbaikan melalui penambahan pupuk.",
                1: "- Tanah cukup mendukung pertumbuhan tanaman namun masih bisa ditingkatkan.<br>"
                   "- Nutrisi cukup stabil untuk produksi pertanian.<br>"
                   "- Perlu pemeliharaan dan pengelolaan rutin.",
                2: "- Tanah memiliki nutrisi optimal dan sangat mendukung pertumbuhan tanaman.<br>"
                   "- Kondisi tanah sangat baik dan produktif.<br>"
                   "- Cocok untuk berbagai jenis tanaman dengan hasil maksimal.",
            }

            st.markdown(f"""
                <div class='metric-card' style='margin-top:14px;'>
                  <div style='font-size:1.1rem; color:#9bb0a4;'>Hasil Prediksi</div>
                  <div style='font-size:1.8rem; font-weight:700; margin:6px 0;'>
                    <span class='pill {color_pill[pred]}'>{LABEL_NAME[pred]}</span>
                  </div>
                  <div style='color:#9bb0a4;'>Keyakinan Model:
                    <b style='color:#e8efe9'>{conf:.2f}%</b>
                  </div>
                  <hr style='border-color:#1f2d27'>
                  <div style='font-weight:600; margin-bottom:6px;'>Deskripsi:</div>
                  <div style='color:#cfd8d2; line-height:1.6;'>{rec[pred]}</div>
                </div>
            """, unsafe_allow_html=True)

            st.write("")
            st.subheader("Probabilitas per Kelas")
            prob_df = pd.DataFrame({
                "Kelas"       : [LABEL_NAME[c] for c in model.classes_],
                "Probabilitas": proba,
            })
            st.bar_chart(prob_df.set_index("Kelas"))
# ============================================================
# PAGE: TENTANG
# ============================================================
def page_tentang():
    st.title("Tentang")
    st.caption("Informasi model, evaluasi, dan referensi proyek")

    # Metrik dari evaluate_model (pakai RF.joblib, tidak training ulang)
    ev = evaluate_model(model)

    st.subheader("Tentang Model")
    c1, c2 = st.columns(2)
    c1.markdown(
        "<div class='metric-card'><div class='label'>Algoritma</div>"
        "<div style='font-size:1.2rem;font-weight:600'>Random Forest Classifier</div></div>",
        unsafe_allow_html=True,
    )
    c2.markdown(
        "<div class='metric-card'><div class='label'>Preprocessing</div>"
        "<div style='font-size:1.2rem;font-weight:600'>StandardScaler + SMOTE (oversampling)</div></div>",
        unsafe_allow_html=True,
    )

    st.subheader("Best Parameters")
    params = [
        "ccp_alpha: 0.001", "criterion: entropy", "max_depth: 12",
        "max_features: sqrt", "max_samples: 0.8", "min_samples_leaf: 5",
        "min_samples_split: 13", "n_estimators: 239",
    ]
    st.markdown(
        "<div style='display:flex; flex-wrap:wrap; gap:10px;'>"
        + "".join([f"<span class='pill pill-green'>{p}</span>" for p in params])
        + "</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Evaluasi Model")
    m1, m2, m3 = st.columns(3)
    for col, (l, v) in zip([m1, m2, m3], [
        ("Test Accuracy",      ev["test_acc"]),
        ("F1 Macro",           ev["f1"]),
        ("Balanced Accuracy",  ev["bal_acc"]),
    ]):
        col.markdown(
            f"<div class='metric-card'><div class='value'>{v*100:.1f}%</div>"
            f"<div class='label'>{l}</div></div>",
            unsafe_allow_html=True,
        )

    st.subheader("Classification Report")
    report = ev["report"]
    rows = []
    for cls in [0, 1, 2]:
        r = report.get(str(cls), {})
        rows.append([
            LABEL_NAME[cls],
            round(r.get("precision", 0), 3),
            round(r.get("recall", 0), 3),
            round(r.get("f1-score", 0), 3),
            int(r.get("support", 0)),
        ])
    st.dataframe(
        pd.DataFrame(rows, columns=["Kelas", "Precision", "Recall", "F1-Score", "Support"]),
        use_container_width=True,
    )

    st.subheader("Confusion Matrix")
    cm     = ev["cm"]
    labels = ["Kurang Subur", "Subur", "Sangat Subur"]
    fig, ax = plt.subplots(figsize=(3, 2.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False,
                linewidths=0.5, linecolor="white", square=True,
                annot_kws={"size": 8, "weight": "bold"}, ax=ax)
    ax.set_xlabel("Predicted", fontsize=7)
    ax.set_ylabel("Actual", fontsize=7)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("Confusion Matrix Model", fontsize=8)
    plt.xticks(rotation=0); plt.yticks(rotation=0)

    col, _ = st.columns([1, 1])
    with col:
        st.pyplot(fig)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style="background:#16201c; padding:15px; border-radius:10px; color:white;">
            <b>Sumber Dataset</b><br><br>
           Sumber dataset yang digunakan berasal dari platform Kaggle yang berisi data terkait kondisi tanah dan berbagai parameter  
           seperti Nitrogen (N), Fosfor (P), Kalium (K), pH, dan unsur mikro lainnya yang mendukung analisis kesuburan tanah. 
           Data ini telah melalui proses pengolahan dan digunakan sebagai dasar dalam membantu menentukan tingkat kesuburan tanah.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#16201c; padding:15px; border-radius:10px; color:white; margin-top:10px;">
            <b>Tentang Saya</b><br><br>
            - Nama : Nurry Nurul Naomi<br>
            - TTL : Purwokerto, 28 Januari 2009<br>
            - Pekerjaan : Pelajar<br>
            - Sekolah : SMKN 1 Purbalingga<br>
            - Email : <a href="mailto:nurry.naomy28@gmail.com">nurry.naomy28@gmail.com</a><br>
            - Instagram : <a href="https://www.instagram.com/nnaoou_/" target="_blank">@nnaoou_</a>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        tools = ["Python", "Scikit-learn", "SMOTE", "Random Forest",
                 "Pandas", "Streamlit", "Numpy", "Matplotlib",
                 "Joblib", "SVM", "KNN", "Decision Tree",
                 "Jupyter Notebook", "Seaborn"]
        tools_html = "".join([f"<span class='pill pill-green'>{t}</span>" for t in tools])
        st.markdown(f"""
        <div style="background:#16201c; padding:15px; border-radius:10px; color:white; margin-bottom:10px;">
            <b>Tools</b><br><br>
            <div style='display:flex; flex-wrap:wrap; gap:8px;'>{tools_html}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#16201c; padding:15px; border-radius:10px; color:white;">
            <b>Latar Belakang</b><br><br>
            Pemilihan topik ini didasarkan pada ketertarikan pencipta terhadap kegiatan berkebun, khususnya dalam mengamati proses bercocok tanam. 
            Selain itu, pengembangan aplikasi ini diharapkan dapat memberikan manfaat bagi pengguna dalam membantu memahami kondisi tanah. 
            Topik ini juga dipilih sebagai bagian dari proyek akhir dalam persiapan Praktik Kerja Lapangan (PKL).
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# ROUTER
# ============================================================
if page == "Beranda":
    page_beranda()
elif page == "Eksplorasi Data":
    page_eksplorasi()
elif page == "Klasifikasi Lahan":
    page_klasifikasi()
else:
    page_tentang()
