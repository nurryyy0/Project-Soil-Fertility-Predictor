# 🌱 SoilSense — Dashboard Klasifikasi Kesuburan Lahan

Dashboard Streamlit untuk klasifikasi kesuburan tanah menggunakan **Random Forest + SMOTE**,
berdasarkan 12 parameter kimia tanah (N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B).

## 📁 Struktur

```
soilsense_streamlit/
├── app.py              # Aplikasi Streamlit (4 halaman)
├── dataset1.csv        # Dataset (880 sampel)
├── requirements.txt
└── README.md
```

## 🚀 Cara menjalankan di VSCode

1. Buka folder `soilsense_streamlit` di VSCode.
2. (Opsional) Buat virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```
5. Browser akan terbuka di `http://localhost:8501`.

## 🧭 Halaman

- **Beranda** — ringkasan dataset, preview & penjelasan fitur.
- **Eksplorasi Data** — distribusi kelas, NPK per kelas, scatter, histogram, heatmap korelasi.
- **Klasifikasi Lahan** — input 12 parameter → prediksi (Kurang Subur / Subur / Sangat Subur) + rekomendasi.
- **Tentang** — info model, evaluasi (akurasi, CV, classification report, confusion matrix).

## 🧠 Model

- Algoritma: `RandomForestClassifier` (`n_estimators=100`, `max_depth=5`, `min_samples_split=5`,
  `min_samples_leaf=2`, `max_features="sqrt"`, `class_weight="balanced"`).
- Preprocessing: `StandardScaler` + SMOTE manual (oversampling kelas minoritas) — diterapkan **hanya di train set**.
- Evaluasi: 5-fold Stratified CV + holdout test 20% data asli.
