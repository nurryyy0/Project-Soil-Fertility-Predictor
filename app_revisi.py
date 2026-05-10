import os
import random
import html
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit.components.v1 as components
import base64
import nbformat
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, balanced_accuracy_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from matplotlib.patches import Patch

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
LABEL_NAME = {0: "Kurang Subur", 1: "Cukup Subur", 2: "Sangat Subur"}

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
# SIDEBAR — Hanya Judul
# ============================================================
def get_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

with st.sidebar:
    logo_b64 = get_image_base64(os.path.join(os.path.dirname(__file__), "ChatGPT Image 10 Mei 2026, 10.30.51.png"))
    st.markdown(f"""
        <div style='text-align:center; padding: 1rem 0 0.5rem;'>
            <img src="data:image/png;base64,{logo_b64}"
                style="width:150px; height:90px; object-fit:contain;
                       display:block; margin:0 auto 8px auto;
                       mix-blend-mode: lighten;">
            <div style='font-weight:700; font-size:1.1rem; color:#4ade80;'>Soil Fertility Predictor</div>
            <div style='font-size:0.75rem; color:#888; margin-top:2px;'>Klasifikasi Kesuburan Tanah</div>
        </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    ## 📖 Petunjuk Penggunaan Aplikasi
    - Pergi ke halaman Klasifikasi Lahan
    - Masukkan semua nilai parameter tanah pada kolom yang tersedia
    - Pastikan data input sesuai
    - Klik tombol prediksi
    - Sistem akan menentukan tingkat kesuburan tanah
    """)
    st.divider()
    st.caption("Machine Learning Project by Nurry")

df    = load_data()
model = load_model() 

st.markdown("""
    <div class="hero">
    <h1>🌿 Smart Soil Fertility Prediction System</h1>
    <p>Sistem cerdas untuk menentukan tingkat kesuburan tanah berdasarkan berbagai
       parameter kimia tanah secara cepat, praktis, dan mudah digunakan.</p>
    </div>
""", unsafe_allow_html=True)

# ============================================================
# TABS NAVIGASI — di bawah judul aplikasi
# ============================================================

st.markdown("""
<style>
.stTabs {
    margin-top: 25px;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1.5px solid #1a3a3a;
    margin-bottom: 15px;
    gap: 25px;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Sora', sans-serif;
    color: #5a8a80;
    font-weight: 500;
    letter-spacing: 0.02em;
    padding: 10px 20px;
    border-radius: 6px 6px 0 0;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    color: #0ef0b8 !important;
    background: rgba(14,240,184,0.06) !important;
    border-bottom: 2.5px solid #0ef0b8 !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #2dd4aa !important;
    background: rgba(14,240,184,0.03) !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #0ef0b8 !important;
    height: 2.5px !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: rgba(6,25,24,0.5);
    border: 0.5px solid #1e3d38;
    border-top: none;
    border-radius: 0 0 10px 10px;
    padding: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

tab1, tab2, tab3,tab4,tab5 = st.tabs(["🏠 Beranda", "📊 Klasifikasi Lahan","🔍 Eksplorasi Data",  "📋 Tentang", "📓 Jupyter Notebook"])

#================================ 
# PAGE BERANDA
#================================
def page_beranda():
    components.html("""
    <!DOCTYPE html>
    <html>
    <head>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                background: transparent;
                font-family: 'Inter', sans-serif;
                color: white;
            }
            .wrapper { padding: 10px 4px 20px 4px; }

            /* Hero */
            .hero-title {
                font-size: 36px;
                font-weight: 800;
                color: white;
                line-height: 1.2;
                margin-bottom: 8px;
            }
            .hero-title span { color: #6fcf97; }
            .hero-subtitle {
                font-size: 14px;
                color: rgba(255,255,255,0.5);
                margin-bottom: 28px;
                line-height: 1.6;
            }

            /* Info cards row */
            .row-2 {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 14px;
                margin-bottom: 28px;
            }
            .info-card {
                background: #1a2e22;
                border: 1px solid rgba(111,207,151,0.15);
                border-radius: 14px;
                padding: 18px 20px;
            }
            .info-card .label {
                font-size: 11px;
                color: rgba(255,255,255,0.4);
                margin-bottom: 6px;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            .info-card .value {
                font-size: 15px;
                font-weight: 600;
                color: white;
            }

            /* Section title */
            .section-title {
                font-size: 22px;
                font-weight: 700;
                color: white;
                margin-bottom: 14px;
            }

            /* Metric cards */
            .metrics-row {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 14px;
                margin-bottom: 28px;
            }
            .metric-card {
                background: #1a2e22;
                border: 1px solid rgba(111,207,151,0.15);
                border-radius: 14px;
                padding: 20px;
            }
            .metric-card .metric-value {
                font-size: 28px;
                font-weight: 800;
                color: white;
                margin-bottom: 4px;
            }
            .metric-card .metric-label {
                font-size: 12px;
                color: rgba(255,255,255,0.45);
            }

            /* Tags */
            .tags-row {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-bottom: 28px;
            }
            .tag {
                background: rgba(111,207,151,0.15);
                color: #6fcf97;
                border: 1px solid rgba(111,207,151,0.3);
                font-size: 12px;
                padding: 5px 14px;
                border-radius: 20px;
                font-weight: 500;
            }

            /* Deskripsi box */
            .desc-box {
                background: #1a2e22;
                border: 1px solid rgba(111,207,151,0.15);
                border-radius: 14px;
                padding: 20px 22px;
                margin-bottom: 16px;
                font-size: 13px;
                color: rgba(255,255,255,0.7);
                line-height: 1.8;
            }

            /* Tujuan list */
            .tujuan-item {
                display: flex;
                align-items: flex-start;
                gap: 12px;
                padding: 14px 0;
                border-bottom: 1px solid rgba(255,255,255,0.06);
            }
            .tujuan-item:last-child { border-bottom: none; }
            .tujuan-icon {
                width: 32px; height: 32px;
                background: rgba(111,207,151,0.15);
                border-radius: 8px;
                display: flex; align-items: center; justify-content: center;
                font-size: 16px;
                flex-shrink: 0;
            }
            .tujuan-text {
                font-size: 13px;
                color: rgba(255,255,255,0.75);
                line-height: 1.6;
                padding-top: 6px;
            }
        </style>
    </head>
    <body>
    <div class="wrapper">

        <!-- Hero -->
        <div style="margin-bottom:28px;">
            <h1 class="hero-title">Prediksi Tingkat<br><span>Kesuburan Tanah</span></h1>
            <a class="hero-subtitle">
                Dukung pertanian yang lebih modern dengan memanfaatkan teknologi untuk mengetahui kondisi tanah secara lebih mudah dan efisien.

            </a>
            <div class="tags-row">
     
            </div>
        </div>

  

        <!-- Tentang Aplikasi -->
        <div class="section-title">Tentang Aplikasi</div>
        <div class="desc-box">
            <a class="hero-subtitle"> Aplikasi berbasis Streamlit yang dirancang untuk memprediksi tingkat kesuburan tanah 
            menggunakan algoritma Machine Learning. Aplikasi ini menganalisis berbagai parameter kimia tanah 
            seperti Nitrogen (N), Fosfor (P), Kalium (K), pH, Electrical Conductivity (EC), Organic Carbon (OC), Sulfur (S), Zinc (Zn), Iron (Fe), Copper (Cu), Manganese (Mn), 
            dan Boron (B) untuk menentukan kategori kesuburan tanah menjadi Kurang Subur, Subur, atau Sangat Subur.
            </a>
        </div>

        <!-- Tujuan -->
        <div class="section-title">Tujuan Aplikasi</div>
        <div class="desc-box" style="padding: 8px 22px;">
            <div class="tujuan-item">
                <div class="tujuan-icon">🌿</div>
                <div class="tujuan-text"><a class="hero-subtitle">Membantu pengguna memahami kondisi kesuburan tanah secara mudah dan cepat.</a></div>
            </div>
            <div class="tujuan-item">
                <div class="tujuan-icon">📊</div>
                <div class="tujuan-text"><a class="hero-subtitle">Memberikan prediksi tingkat kesuburan berdasarkan parameter tanah yang diinputkan.</a></div>
            </div>
            <div class="tujuan-item">
                <div class="tujuan-icon">📇</div>
                <div class="tujuan-text"><a class="hero-subtitle">Menyediakan visualisasi data agar informasi lebih mudah dipahami pengguna.</a></div>
            </div>
            <div class="tujuan-item">
                <div class="tujuan-icon">📈</div>
                <div class="tujuan-text"><a class="hero-subtitle">Membantu meningkatkan produktivitas pertanian melalui analisis tanah yang lebih akurat.</a></div>
            </div>
            <div class="tujuan-item">
                <div class="tujuan-icon">🪴</div>
                <div class="tujuan-text"><a class="hero-subtitle">Meningkatkan efisiensi pengambilan keputusan terkait pemupukan dan pengelolaan lahan.</a></div>
            </div>
        </div>

    </div>
    </body>
    </html>
    """, height=740)
# ============================================================
# PAGE: KLASIFIKASI LAHAN
# ============================================================
from fpdf import FPDF  # pip install fpdf2
import io
from datetime import datetime

def buat_pdf_laporan(predicted_label, conf, warna_hex, deskripsi,
                     rows, kurang, berlebih, prob_df, fig):
    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def hex2rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    # ── HEADER ────────────────────────────────────────────────────────────────
    pdf.set_fill_color(30, 42, 39)
    pdf.rect(0, 0, 210, 28, "F")
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.set_y(8)
    pdf.cell(0, 10, "Laporan Prediksi Kesuburan Tanah", ln=True, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(180, 200, 190)
    pdf.cell(0, 6, f"Digenerate: {datetime.now().strftime('%d %B %Y  %H:%M')}", ln=True, align="C")
    pdf.ln(6)

    # ── 1. HASIL PREDIKSI ─────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(60, 60, 60)
    pdf.set_fill_color(240, 245, 242)
    pdf.cell(0, 8, "  1. Hasil Prediksi", ln=True, fill=True)
    pdf.ln(3)

    r, g, b = hex2rgb(warna_hex)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(r, g, b)
    pdf.cell(0, 12, predicted_label, ln=True, align="C")

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 7, f"Keyakinan Model: {conf:.2f}%", ln=True, align="C")
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 7, "Deskripsi:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(70, 70, 70)
    for baris in deskripsi.replace("<br>", "\n").split("\n"):
        baris = baris.strip().lstrip("-").strip()
        if baris:
            pdf.set_x(15)
            pdf.multi_cell(0, 6, f"  -  {baris}")
    pdf.ln(4)

    # ── 2. GRAFIK PROBABILITAS ────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(60, 60, 60)
    pdf.set_fill_color(240, 245, 242)
    pdf.cell(0, 8, "  2. Probabilitas per Kelas", ln=True, fill=True)
    pdf.ln(3)

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png", dpi=120, bbox_inches="tight", facecolor="#FFFFFF")
    img_buf.seek(0)
    pdf.image(img_buf, x=30, w=150)
    pdf.ln(4)

    # ── 3. TABEL PERBANDINGAN ─────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(240, 245, 242)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 8, "  3. Perbandingan Input vs Data Ideal", ln=True, fill=True)
    pdf.ln(3)

    col_w  = [52, 38, 38, 28, 34]
    headers = ["Fitur", "Nilai Input", "Nilai Ideal", "Selisih", "Status"]
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(50, 80, 65)
    pdf.set_text_color(255, 255, 255)
    for w, h in zip(col_w, headers):
        pdf.cell(w, 7, h, border=1, fill=True, align="C")
    pdf.ln()

    STATUS_COLOR = {
        " Optimal" : (220, 245, 220),
        " Kurang"  : (255, 230, 230),
        " Berlebih": (255, 240, 210),
    }
    pdf.set_font("Helvetica", "", 9)
    for row in rows:
        vals   = [row["Fitur"], row["Nilai Input"], row["Nilai Ideal"],
                  row["Selisih"], row["Status"]]
        r2, g2, b2 = STATUS_COLOR.get(row["Status"], (245, 245, 245))
        pdf.set_fill_color(r2, g2, b2)
        pdf.set_text_color(40, 40, 40)
        for w, v in zip(col_w, vals):
            pdf.cell(w, 6, str(v), border=1, fill=True, align="C")
        pdf.ln()
    pdf.ln(4)

    # ── 4. REKOMENDASI ────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(240, 245, 242)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 8, "  4. Rekomendasi Perbaikan Lahan", ln=True, fill=True)
    pdf.ln(3)

    if not kurang and not berlebih:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(50, 150, 80)
        pdf.cell(0, 8, "Semua parameter sudah optimal!", ln=True, align="C")
    else:
        if kurang:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(229, 57, 53)
            pdf.cell(0, 7, "Parameter yang Perlu Ditingkatkan:", ln=True)
            for no, item in enumerate(kurang, 1):
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(50, 50, 50)
                pdf.set_x(20)
                pdf.cell(0, 6, f"{no}. {item['fitur']}", ln=True)

                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(80, 80, 80)

                pdf.set_x(25)
                pdf.multi_cell(160, 6, item['saran'])

                pdf.ln(2)

        if berlebih:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(255, 167, 38)
            pdf.cell(0, 7, "Parameter yang Perlu Dikurangi:", ln=True)
            for no, item in enumerate(berlebih, 1):
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(50, 50, 50)
                pdf.set_x(20)
                pdf.cell(0, 6, f"{no}. {item['fitur']}", ln=True)

                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(80, 80, 80)

                pdf.set_x(25)
                pdf.multi_cell(160, 6, item['saran'])

    return bytes(pdf.output())

#============================
# PAGE KLASIFIKASI 
#============================
def page_klasifikasi():
    st.title("Klasifikasi Lahan")
    st.subheader("Masukkan parameter tanah untuk memprediksi tingkat kesuburan")

    features = df.drop("Output", axis=1).columns.tolist()
    ranges   = {f: (float(df[f].min()), float(df[f].max())) for f in features}
    if "reset_counter" not in st.session_state:
        st.session_state.reset_counter = 0

    default_values = {
        "N":  264,  "P":  12.1, "K":  560,  "pH": 7.9,
        "EC": 0.63, "OC": 0.39, "S":  4.22, "Zn": 0.33,
        "Fe": 3.22, "Cu": 0.75, "Mn": 11.60,"B":  0.30,
    }

    NAMA_FITUR = {
        "N":  "Nitrogen (N)",           "P":  "Fosfor (P)",
        "K":  "Kalium (K)",             "pH": "pH Tanah",
        "EC": "Electrical Conductivity","OC": "Organic Carbon (OC)",
        "S":  "Sulfur (S)",             "Zn": "Zinc (Zn)",
        "Fe": "Iron (Fe)",              "Cu": "Copper (Cu)",
        "Mn": "Manganese (Mn)",         "B":  "Boron (B)",
    }

    SATUAN = {
        "N":"mg/kg","P":"mg/kg","K":"mg/kg","pH":"",
        "EC":"dS/m","OC":"%",  "S":"mg/kg","Zn":"mg/kg",
        "Fe":"mg/kg","Cu":"mg/kg","Mn":"mg/kg","B":"mg/kg",
    }

    DATA_IDEAL = {
        "N":314.0,"P":83.3, "K":486.0,"pH":7.41,
        "EC":0.80,"OC":0.68,"S":7.54, "Zn":0.51,
        "Fe":7.63,"Cu":1.36,"Mn":12.06,"B":0.25,
    }

    SARAN_FITUR = {
        "N" :("Tambahkan pupuk Urea atau ZA untuk meningkatkan Nitrogen.",
              "Kurangi pupuk nitrogen, hindari kelebihan yang menyebabkan lodging."),
        "P" :("Tambahkan pupuk SP-36 atau TSP untuk meningkatkan Fosfor.",
              "Kurangi pupuk fosfat, kelebihan P menghambat penyerapan Zn dan Fe."),
        "K" :("Tambahkan pupuk KCl atau K2SO4 untuk meningkatkan Kalium.",
              "Kurangi pupuk kalium, kelebihan K mengganggu penyerapan Mg dan Ca."),
        "pH":("Lakukan pengapuran dengan dolomit untuk menaikkan pH tanah.",
              "Tambahkan belerang atau bahan organik asam untuk menurunkan pH."),
        "EC":("Perbaiki struktur tanah dan tambahkan bahan organik untuk meningkatkan EC.",
              "Kurangi pemupukan dan lakukan pencucian tanah untuk menurunkan EC."),
        "OC":("Tambahkan kompos atau pupuk kandang untuk meningkatkan bahan organik.",
              "Kurangi pengolahan tanah berlebihan agar bahan organik tetap terjaga."),
        "S" :("Tambahkan pupuk ZA atau gipsum untuk meningkatkan kandungan Sulfur.",
              "Kurangi pupuk berbasis sulfur agar tidak terjadi kelebihan."),
        "Zn":("Semprotkan Zinc Sulfate (ZnSO4) untuk mengatasi defisiensi Zinc.",
              "Kurangi aplikasi Zinc, kelebihan dapat menghambat penyerapan Fe dan Mn."),
        "Fe":("Tambahkan chelated iron (Fe-EDTA) atau pupuk mikro mengandung besi.",
              "Perbaiki drainase tanah karena kelebihan Fe biasanya pada lahan tergenang."),
        "Cu":("Semprotkan Copper Sulfate (CuSO4) untuk mengatasi defisiensi tembaga.",
              "Kurangi aplikasi tembaga, kelebihan Cu bersifat toksik bagi tanaman."),
        "Mn":("Tambahkan Manganese Sulfate untuk meningkatkan kadar Mangan.",
              "Perbaiki drainase dan naikkan pH tanah untuk mengurangi ketersediaan Mn."),
        "B" :("Semprotkan Borax atau pupuk mikro mengandung Boron.",
              "Kurangi aplikasi Boron, rentang optimal sangat sempit dan mudah toksik."),
    }

    WARNA_KELAS = {
        "Kurang Subur":"#E53935",
        "Cukup Subur": "#00ACC1",
        "Sangat Subur":"#43A047",
    }

    # ── Input Form ────────────────────────────────────────────────────────────
    counter = st.session_state.reset_counter
    cols    = st.columns(4)
    inputs  = {}
    for i, f in enumerate(features):
        lo, hi = ranges[f]
        with cols[i % 4]:
            inputs[f] = st.number_input(
                NAMA_FITUR.get(f, f),
                value=float(default_values.get(f, (lo + hi) / 2)),
                step=0.1,
                format="%.2f",
                placeholder="...",
                key=f"input_{f}_{counter}"
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

            predicted_label = LABEL_NAME[pred]
            warna           = WARNA_KELAS[predicted_label]

            DESKRIPSI = {
                "Kurang Subur": (
                    "- Kandungan unsur hara sangat rendah, tidak mencukupi kebutuhan tanaman.<br>"
                    "- Struktur tanah buruk dan pH tidak seimbang.<br>"
                    "- Memerlukan perbaikan menyeluruh sebelum dapat ditanami."
                ),
                "Cukup Subur": (
                    "- Kandungan unsur hara cukup untuk mendukung pertumbuhan tanaman.<br>"
                    "- Kondisi tanah sudah layak tanam dengan pemeliharaan rutin.<br>"
                    "- Produktivitas dapat ditingkatkan dengan pemupukan berimbang."
                ),
                "Sangat Subur": (
                    "- Kandungan unsur hara lengkap dan optimal untuk pertumbuhan tanaman.<br>"
                    "- Struktur tanah baik, pH seimbang, dan mikronutrien mencukupi.<br>"
                    "- Lahan siap untuk produksi pertanian intensif."
                ),
            }

            # ── Hasil Prediksi Card ───────────────────────────────────────────
            st.write(f"""
                <div class='metric-card' style='margin-top:14px;'>
                  <div style='font-size:1.1rem; color:#9bb0a4;'>Hasil Prediksi</div>
                  <div style='font-size:2.8rem; font-weight:800; margin:10px 0; color:{warna};
                              letter-spacing:0.5px;'>
                    {predicted_label}
                  </div>
                  <div style='color:#9bb0a4;'>Keyakinan Model:
                    <b style='color:#e8efe9; font-size:1.15rem;'>{conf:.2f}%</b>
                  </div>
                  <hr style='border-color:#1f2d27'>
                  <div style='font-weight:600; margin-bottom:6px;'>Deskripsi:</div>
                  <div style='color:#cfd8d2; line-height:1.6;'>{DESKRIPSI[predicted_label]}</div>
                </div>
            """, unsafe_allow_html=True)

            # ── Bar Chart Probabilitas ────────────────────────────────────────
            st.write("")
            st.subheader("Probabilitas per Kelas")

            prob_df = pd.DataFrame({
                "Kelas"       : [LABEL_NAME[c] for c in model.classes_],
                "Probabilitas": proba,
            })

            bar_colors = ["#E53935", "#00ACC1", "#43A047"]

            fig, ax = plt.subplots(figsize=(5, 3))
            fig.patch.set_facecolor("#1E1E2E")
            ax.set_facecolor("#2A2A3E")

            bars = ax.bar(
                prob_df["Kelas"], prob_df["Probabilitas"],
                color=bar_colors[:len(prob_df)],
                edgecolor="white", linewidth=0.5, width=0.5
            )

            for bar, val in zip(bars, prob_df["Probabilitas"]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center", va="bottom", color="white", fontsize=8
                )

            ax.set_ylim(0, 1.1)
            ax.set_ylabel("Probabilitas", color="white", fontsize=9)
            ax.set_xlabel("Kelas", color="white", fontsize=9)
            ax.tick_params(colors="white", labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#555577")
            ax.yaxis.grid(True, color="#3A3A5A", linewidth=0.5)
            ax.set_axisbelow(True)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)

            # ── Perbandingan Input vs Ideal ───────────────────────────────────
            st.markdown("---")
            st.subheader("📊 Perbandingan Input vs Data Ideal")

            TOLERANSI_ABSOLUT = {
                "N":  10,  "P":  10,  "K":  10,
                "pH": 1.0, "EC": 0.1, "OC": 0.1,
                "S":  1.0, "Zn": 0.1, "Fe": 1.0,
                "Cu": 1.0, "Mn": 1.0, "B":  0.1,
            }

            rows     = []
            kurang   = []
            berlebih = []

            for fitur, ideal_val in DATA_IDEAL.items():
                input_val = float(inputs[fitur])
                selisih   = input_val - ideal_val
                toleransi = TOLERANSI_ABSOLUT.get(fitur, ideal_val * 0.10)

                if abs(selisih) <= toleransi:
                    status = " Optimal"
                elif selisih < 0:
                    status = " Kurang"
                    kurang.append({
                        "fitur": NAMA_FITUR[fitur],
                        "saran": SARAN_FITUR[fitur][0],
                    })
                else:
                    status = "Berlebih"
                    berlebih.append({
                        "fitur": NAMA_FITUR[fitur],
                        "saran": SARAN_FITUR[fitur][1],
                    })
                rows.append({
                    "Fitur"      : NAMA_FITUR[fitur],
                    "Nilai Input": f"{input_val:.2f} {SATUAN[fitur]}".strip(),
                    "Nilai Ideal": f"{ideal_val:.2f} {SATUAN[fitur]}".strip(),
                    "Selisih"    : f"{selisih:+.2f}",
                    "Status"     : status,
                })

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

            # ── Rekomendasi Spesifik ──────────────────────────────────────────
            st.markdown("---")
            st.subheader("💡 Rekomendasi Perbaikan Lahan")

            if not kurang and not berlebih:
                st.success("🎉 Semua parameter sudah optimal! Lahan kamu dalam kondisi terbaik.")
            else:
                if kurang:
                    items = ''.join(
                        f"<li><b style='color:#FFFFFF'>{r['fitur']}</b>"
                        f"<br><span style='color:#BBBBBB'>{r['saran']}</span></li>"
                        for r in kurang
                    )
                    st.markdown(f"""
                        <div style="background:#1E1E2E; border-left:5px solid #E53935;
                                    border-radius:8px; padding:16px 20px; margin-bottom:12px;">
                            <p style="color:#E53935; font-weight:bold; margin:0 0 8px 0;">
                                ⬇️ Parameter yang Perlu Ditingkatkan
                            </p>
                            <ol style="color:#DDDDDD; margin:0; padding-left:18px; line-height:2.2;">
                                {items}
                            </ol>
                        </div>
                    """, unsafe_allow_html=True)

                if berlebih:
                    items = ''.join(
                        f"<li><b style='color:#FFFFFF'>{r['fitur']}</b>"
                        f"<br><span style='color:#BBBBBB'>{r['saran']}</span></li>"
                        for r in berlebih
                    )
                    st.markdown(f"""
                        <div style="background:#1E1E2E; border-left:5px solid #FFA726;
                                    border-radius:8px; padding:16px 20px; margin-bottom:12px;">
                            <p style="color:#FFA726; font-weight:bold; margin:0 0 8px 0;">
                                ⬆️ Parameter yang Perlu Dikurangi
                            </p>
                            <ol style="color:#DDDDDD; margin:0; padding-left:18px; line-height:2.2;">
                                {items}
                            </ol>
                        </div>
                    """, unsafe_allow_html=True)

            # ── Download PDF (setelah semua rekomendasi) ──────────────────────
            st.markdown("---")
            pdf_bytes = buat_pdf_laporan(
                predicted_label = predicted_label,
                conf            = conf,
                warna_hex       = warna,
                deskripsi       = DESKRIPSI[predicted_label],
                rows            = rows,
                kurang          = kurang,
                berlebih        = berlebih,
                prob_df         = prob_df,
                fig             = fig,
            )
            st.download_button(
                label              = "📄 Download Laporan PDF",
                data               = pdf_bytes,
                file_name          = "laporan_kesuburan_tanah.pdf",
                mime               = "application/pdf",
                use_container_width= True,
                type               = "primary",
            )
                   

# ============================================================
# PAGE: BERANDA & EKSPLORASI DATA
# ============================================================
def page_eksplorasi():
    GREEN  = "#2fa05a"
    ORANGE = "#e8a13a"
    RED    = "#d96a6a"

    st.title("Dataset")
    st.write(f"Dataset ini berisi **{len(df)} sampel** tanah dengan **{df.shape[1]-1} parameter kimia** "
             f"yang digunakan untuk mengklasifikasikan tingkat kesuburan lahan menjadi 3 kelas:")

    c1, c2, c3, c4,c5 = st.columns(5)
    counts = df["Output"].value_counts()
    cards = [
        ("Jumlah Sampel",  len(df)),
        ("Jumlah Fitur",   df.shape[1] - 1),
        ("Jumlah Kelas",   df["Output"].nunique()),
        ("Missing Values", int(df.isna().sum().sum())),
        ("duplicated Values", int(df.duplicated().sum())),

    ]
    for col, (label, val) in zip([c1, c2, c3, c4,c5], cards):
        col.markdown(
            f"<div class='metric-card'><div class='value'>{val}</div>"
            f"<div class='label'>{label}</div></div>",
            unsafe_allow_html=True,
        )
    st.write("   ")
    st.markdown(
        f"<span class='pill pill-red'>Kurang Subur ({counts.get(0,0)} sampel)</span>"
        f"<span class='pill pill-yellow'>Cukup Subur ({counts.get(1,0)} sampel)</span>"
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
    st.write("")
    st.divider()
    st.title("Eksplorasi Data")
    st.write("Visualisasi dan analisis karakteristik dataset kesuburan tanah")

    st.subheader("📊 Distribusi Fitur")

    nama_fitur = {
        "N"  : "Nitrogen (N)",
        "P"  : "Fosfor (P)",
        "K"  : "Kalium (K)",
        "pH" : "pH Tanah",
        "EC" : "Electrical Conductivity (EC)",
        "OC" : "Organic Carbon (OC)",
        "S"  : "Sulfur (S)",
        "Zn" : "Seng (Zn)",
        "Fe" : "Besi (Fe)",
        "Cu" : "Tembaga (Cu)",
        "Mn" : "Mangan (Mn)",
        "B"  : "Boron (B)",
    }

    fitur_list   = [col for col in df.columns if col != 'Output']
    fitur_dipilih = st.selectbox(
        "Pilih parameter tanah:",
        fitur_list,
        format_func=lambda x: nama_fitur.get(x, x)
    )

    fig, ax = plt.subplots(figsize=(6, 3))

    ax.hist(df[fitur_dipilih], bins=20, color="#2e7d32", edgecolor="white", alpha=0.85)
    nama_dipilih = nama_fitur.get(fitur_dipilih, fitur_dipilih)

    ax.set_title(f"Distribusi {nama_dipilih}", fontsize=14, fontweight="bold", color="white")
    ax.set_xlabel(nama_dipilih, fontsize=11, color="white")
    ax.set_ylabel("Jumlah Data", fontsize=11, color="white")

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_facecolor("#1e2e22")
    fig.patch.set_facecolor("#16201c")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor((1, 1, 1, 0.1))

    st.pyplot(fig)
    col1, col2, col3, col4 = st.columns(4)

    def metric_box(col, label, value):
        col.markdown(
            f"""<div style="background:#1e2e22; border:0.5px solid #2a3a33;
                border-radius:10px; padding:17px 19px; text-align:center;">
                <div style="font-size:12px; color:rgba(255,255,255,0.5);
                    margin-bottom:6px;font-weight:500;">{label}</div>
                <div style="font-size:33px; font-weight:600;
                    color:white;">{value}</div>
            </div>""",
            unsafe_allow_html=True
        )

    metric_box(col1, "Rata-rata", f"{df[fitur_dipilih].mean():.2f}")
    metric_box(col2, "Median",    f"{df[fitur_dipilih].median():.2f}")
    metric_box(col3, "Min",       f"{df[fitur_dipilih].min():.2f}")
    metric_box(col4, "Max",       f"{df[fitur_dipilih].max():.2f}")

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
        ax.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig)

    st.subheader("🔍 Scatter Plot Antar Fitur")

    fitur_list = [col for col in df.columns if col != "Output"]

    col1, col2 = st.columns(2)
    with col1:
        sumbu_x = st.selectbox("Sumbu X", fitur_list, index=0,
                                format_func=lambda x: nama_fitur.get(x, x))
    with col2:
        sumbu_y = st.selectbox("Sumbu Y", fitur_list, index=1,
                                format_func=lambda x: nama_fitur.get(x, x))

    nama_x = nama_fitur.get(sumbu_x, sumbu_x)
    nama_y = nama_fitur.get(sumbu_y, sumbu_y)

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#0f1715")
    ax.set_facecolor("#16201c")

    df["Output_label"] = df["Output"].map(LABEL_NAME)

    warna_kelas = {
        "Kurang Subur" : "#d96a6a",
        "Cukup Subur"  : "#e8a13a",
        "Sangat Subur" : "#2fa05a",
    }

    for kelas, grup in df.groupby("Output_label"):
        ax.scatter(
            grup[sumbu_x], grup[sumbu_y],
            label=kelas,
            color=warna_kelas[kelas],
            alpha=0.7, s=50, edgecolors="none"
        )

    ax.set_xlabel(nama_x, color="#cfd8d2", fontsize=11)
    ax.set_ylabel(nama_y, color="#cfd8d2", fontsize=11)
    ax.set_title(f"{nama_x} vs {nama_y}", fontsize=13,
                fontweight="bold", color="#e8efe9", pad=14)
    ax.tick_params(colors="#cfd8d2")
    ax.spines[:].set_color("#2a3a33")
    ax.legend(title="Kelas", facecolor="#16201c",
            edgecolor="#2a3a33", labelcolor="#cfd8d2",
            title_fontsize=9, fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)



    st.subheader("🌱 Fitur Paling Berpengaruh terhadap Prediksi Kesuburan Tanah")

    features    = df.drop("Output", axis=1).columns.tolist()[:12]
    rf          = model.named_steps["rf"]
    importances = rf.feature_importances_


    fi_df = pd.DataFrame({
        "Fitur"      : features,
        "Importance" : importances
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    fi_df.index += 1
    fi_df.index.name = "Rank"

    fi_plot = fi_df.sort_values("Importance", ascending=True)

    q33 = fi_plot["Importance"].quantile(0.33)
    q66 = fi_plot["Importance"].quantile(0.66)



    colors = [
        GREEN  if v >= q66 else
        ORANGE if v >= q33 else
        RED
        for v in fi_plot["Importance"]
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#0f1715")
    ax.set_facecolor("#16201c")

    bars = ax.barh(fi_plot["Fitur"], fi_plot["Importance"],
                color=colors, edgecolor="none", height=0.6)

    for bar, val in zip(bars, fi_plot["Importance"]):
        ax.text(bar.get_width() + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, color="#cfd8d2")

    ax.set_xlabel("Importance Score", color="#cfd8d2")
    ax.set_title("Feature Importance — Random Forest", fontsize=13,
                fontweight="bold", color="#e8efe9", pad=14)
    ax.set_xlim(0, fi_plot["Importance"].max() + 0.06)
    ax.tick_params(colors="#cfd8d2")
    ax.spines[:].set_color("#2a3a33")

    legend = [
        Patch(color=GREEN,  label="High importance"),
        Patch(color=ORANGE, label="Mid importance"),
        Patch(color=RED,    label="Low importance"),
    ]
    ax.legend(handles=legend, loc="lower right",
            facecolor="#16201c", edgecolor="#2a3a33",
            labelcolor="#cfd8d2", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)

    st.caption(f"✅ Paling berpengaruh: **{fi_df.iloc[0]['Fitur']}** ({fi_df.iloc[0]['Importance']:.4f}) &nbsp;|&nbsp; ⚠️ Paling lemah: **{fi_df.iloc[-1]['Fitur']}** ({fi_df.iloc[-1]['Importance']:.4f})")
    
# ============================================================
# PAGE: TENTANG
# ============================================================
def page_tentang():
    st.title("Tentang model")
    st.write("Informasi model, evaluasi, dan referensi proyek")

  
    ev = evaluate_model(model)


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
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("Confusion Matrix Model", fontsize=8)
    plt.xticks(rotation=0); plt.yticks(rotation=0)

    col, _ = st.columns([1, 1])
    with col:
        st.pyplot(fig)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Sumber Dataset")
        st.markdown("""
        <div style="background:#16201c; padding:15px; border-radius:10px; color:white;">
           Sumber dataset yang digunakan berasal dari platform Kaggle yang berisi data terkait kondisi tanah dan berbagai parameter  
           seperti Nitrogen (N), Fosfor (P), Kalium (K), pH, dan unsur mikro lainnya yang mendukung analisis kesuburan tanah. 
           Data ini telah melalui proses pengolahan dan digunakan sebagai dasar dalam membantu menentukan tingkat kesuburan tanah.
        </div>
        """, unsafe_allow_html=True)


    with c2:
        tools = ["Python", "Scikit-learn", "SMOTE", "Random Forest",
                 "Pandas", "Streamlit", "Numpy", "Matplotlib",
                 "Joblib", "SVM", "KNN", "Decision Tree",
                 "Jupyter Notebook", "Seaborn"]
        tools_html = "".join([f"<span class='pill pill-green'>{t}</span>" for t in tools])
        st.subheader("Tools")
        st.markdown(f"""
        <div style="background:#16201c; padding:15px; border-radius:10px; color:white; margin-bottom:10px;">
            <div style='display:flex; flex-wrap:wrap; gap8px;'>{tools_html}</div>
        </div>
        """, unsafe_allow_html=True)
    st.write("")
    st.divider()


    st.markdown(
        "<h1>Tentang Saya</h1>",
        unsafe_allow_html=True
    )

    
    def load_image_b64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    img_b64 = load_image_b64("WhatsApp Image 2026-05-04 at 10.26.55.jpeg")


    st.subheader("Developer Streamlit App")

    components.html(f"""
    <div style="background:#16201c; padding:28px; border-radius:16px; color:white; font-family:sans-serif;">

        <!-- Foto & Identitas -->
        <div style="text-align:center; margin-bottom:24px;">
            <img src="data:image/jpeg;base64,{img_b64}"
                style="width:110px; height:110px; border-radius:50%;
                    object-fit:cover; object-position:top;
                    border:2px solid rgba(255,255,255,0.2);
                    display:block; margin:0 auto 14px auto;">

            <div style="font-size:22px; font-weight:600; margin-bottom:6px;">Nurry Nurul Naomi</div>
            <div style="font-size:15px; color:rgba(255,255,255,0.55); margin-bottom:12px;">Streamlit Developer</div>

            <div style="display:flex; justify-content:center; gap:8px; flex-wrap:wrap;">
                <span style="background:#1a3a5c; color:#7ab8f5; font-size:13px; padding:4px 16px; border-radius:20px;">RPL</span>
                <span style="background:#1e2e22; color:rgba(255,255,255,0.75); font-size:13px; padding:4px 16px; border-radius:20px;">SMKN 1 Purbalingga</span>
            </div>
        </div>

        <!-- Tanggal Lahir -->
        <div style="display:flex; align-items:center; gap:16px;
            background:#1e2e22; border-radius:10px; padding:14px 16px; margin-bottom:12px;">
            <span style="font-size:24px;">📅</span>
            <div>
                <div style="font-size:12px; color:rgba(255,255,255,0.5); margin-bottom:4px;">Tanggal Lahir</div>
                <div style="font-size:16px; font-weight:500;">Purwokerto, 28 Januari 2009</div>
            </div>
        </div>

        <!-- Email -->
        <div style="display:flex; align-items:center; gap:16px;
            background:#1e2e22; border-radius:10px; padding:14px 16px; margin-bottom:12px;">
            <span style="font-size:24px;">✉️</span>
            <div>
                <div style="font-size:12px; color:rgba(255,255,255,0.5); margin-bottom:4px;">Email</div>
                <a href="mailto:nurry.naomy28@gmail.com"
                style="font-size:16px; font-weight:500; color:#7ab8f5; text-decoration:none;">
                    nurry.naomy28@gmail.com
                </a>
            </div>
        </div>

        <!-- GitHub & Instagram -->
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
            <a href="https://github.com/nurryyy0" target="_blank" style="
                display:flex; align-items:center; gap:14px;
                background:#1e2e22; border-radius:10px; padding:14px 16px; text-decoration:none;">
                <span style="font-size:24px;">🐙</span>
                <div>
                    <div style="font-size:12px; color:rgba(255,255,255,0.5); margin-bottom:4px;">GitHub</div>
                    <div style="font-size:15px; font-weight:500; color:white;">@nurryyy0</div>
                </div>
            </a>
            <a href="https://www.instagram.com/nnaoou_/" target="_blank" style="
                display:flex; align-items:center; gap:14px;
                background:#1e2e22; border-radius:10px; padding:14px 16px; text-decoration:none;">
                <span style="font-size:24px;">📸</span>
                <div>
                    <div style="font-size:12px; color:rgba(255,255,255,0.5); margin-bottom:4px;">Instagram</div>
                    <div style="font-size:15px; font-weight:500; color:white;">@nnaoou_</div>
                </div>
            </a>
        </div>

    </div>
    """, height=540)

#===========================
# JUPYTER KODE
#==========================
def page_jupyter():
    st.title("📓 Notebook Cell Viewer")

    nb_path = os.path.join(os.path.dirname(__file__), "KLASIFIKASI_LAHAN.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    for i, cell in enumerate(nb.cells):

        # ── Header cell ──────────────────────────────────────
        tipe  = cell.cell_type
        label = "🟦 Markdown" if tipe == "markdown" else "🟩 Code"
        st.markdown(
            f"""<div style="background:#1e2e22; border-left:3px solid #2fa05a;
                padding:6px 14px; border-radius:6px; margin-bottom:4px;">
                <span style="color:#7ab8f5; font-size:12px; font-weight:600;">
                Cell [{i+1}]</span>
                <span style="color:rgba(255,255,255,0.4); font-size:12px;">
                &nbsp;·&nbsp;{label}</span>
            </div>""",
            unsafe_allow_html=True
        )

        # ── Isi cell ─────────────────────────────────────────
        if tipe == "markdown":
            st.markdown(cell.source)

        elif tipe == "code":
            st.code(cell.source, language="python")

            # ── Output cell ───────────────────────────────────
            for output in cell.get("outputs", []):

                # Teks / print
                if output.output_type == "stream":
                    teks = html.escape(output.text)
                    st.markdown(
                        f"""<div style="background:#0f1715; border-radius:6px;
                            padding:10px 14px; font-family:monospace;
                            font-size:13px; color:#a8d5b5;
                            white-space:pre-wrap; margin-top:4px;">{teks}</div>""",
                        unsafe_allow_html=True
                    )

                elif output.output_type in ("display_data", "execute_result"):

                    # Gambar
                    if "image/png" in output.data:
                        import base64
                        img_data = output.data["image/png"]
                        st.markdown(
                            f'<img src="data:image/png;base64,{img_data}" '
                            f'style="max-width:100%; border-radius:8px; margin-top:6px;">',
                            unsafe_allow_html=True
                        )

                    # Tabel / teks
                    elif "text/html" in output.data:
                        st.markdown(output.data["text/html"], unsafe_allow_html=True)

                    elif "text/plain" in output.data:
                        teks = html.escape(output.data["text/plain"])
                        st.markdown(
                            f"""<div style="background:#0f1715; border-radius:6px;
                                padding:10px 14px; font-family:monospace;
                                font-size:13px; color:#a8d5b5;
                                white-space:pre-wrap; margin-top:4px;">{teks}</div>""",
                            unsafe_allow_html=True
                        )

        st.markdown("<hr style='border:0.5px solid #1e2e22; margin:12px 0'>",
                    unsafe_allow_html=True)
# ============================================================
# ROUTER — menggunakan tabs
# ============================================================
with tab1:
    page_beranda()
with tab2:
    page_klasifikasi()

with tab3:
    page_eksplorasi()

with tab4:
    page_tentang()

with tab5:
    page_jupyter()