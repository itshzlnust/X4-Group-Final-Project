# 📋 DOKUMENTASI PROYEK — Kelompok X4
## Intelligent Steel Surface Fault Detection

> **Tugas Besar Kecerdasan Buatan (TE2514031)**  
> Institut Teknologi Kalimantan | Semester Gasal 2026/2027  
> **Dosen:** Happy Aprillia, S.ST., M.T., M.Eng., Ph.D.

---

## 👥 Anggota Kelompok

| NIM | Nama |
|-----|------|
| 11231084 | Rahmahdaniah Fitri |
| 17231008 | Aulia Wahdah Atika Rahmah |
| 10231042 | Ilham Ahmad Fahriji |

---

## 📁 Struktur File Proyek

```
kelompok_x4_steel_fault/
│
├── kelompok_x4_hybrid_pipeline.py       ← Pipeline utama (GA → PSO → MLP + 1D-CNN → Fuzzy)
├── streamlit_dashboard.py               ← Dashboard visualisasi 7 halaman
├── Kelompok_X4_Steel_Fault_Hybrid_AI_FIXED.ipynb  ← Jupyter Notebook
├── pipeline_artifacts.pkl               ← Hasil training tersimpan (model + data)
├── DOKUMENTASI.md                       ← File ini
├── README.md                            ← Panduan singkat
│
└── data/
    ├── Faults.NNA                       ← Data utama (1941 baris × 34 kolom)
    └── Faults27x7_var                   ← Nama variabel (27 fitur + 7 kelas)
```

---

## 🗄️ Dataset

### Informasi Umum

| Atribut | Detail |
|---------|--------|
| Nama | Steel Plates Faults Dataset |
| Sumber | UCI Machine Learning Repository (ID: 198) |
| Jumlah Sampel | **1.941 sampel** |
| Jumlah Fitur | **27 fitur** (14 integer + 13 kontinu) |
| Jumlah Kelas Target | **7 kelas** cacat |
| File Lokal | `data/Faults.NNA` + `data/Faults27x7_var` |

### Format File Dataset

**`data/Faults.NNA`** — File data utama:
- Format: space-separated values, tanpa header
- Dimensi: 1941 baris × 34 kolom
- 27 kolom pertama = fitur, 7 kolom terakhir = one-hot encoding kelas

**`data/Faults27x7_var`** — File nama variabel:
- Daftar 34 nama variabel (27 fitur + 7 kelas target)
- Dibaca satu per baris, dipakai sebagai nama kolom

### Fitur-Fitur Dataset

#### Fitur Integer (14 fitur)

| No | Nama Fitur | Deskripsi |
|----|-----------|-----------|
| 1 | `X_Minimum` | Koordinat X minimum cacat pada permukaan baja |
| 2 | `X_Maximum` | Koordinat X maksimum cacat pada permukaan baja |
| 3 | `Y_Minimum` | Koordinat Y minimum cacat (posisi sepanjang conveyor) |
| 4 | `Y_Maximum` | Koordinat Y maksimum cacat (posisi sepanjang conveyor) |
| 5 | `Pixels_Areas` | Luas area cacat dalam piksel |
| 6 | `X_Perimeter` | Keliling cacat pada arah X |
| 7 | `Y_Perimeter` | Keliling cacat pada arah Y |
| 8 | `Sum_of_Luminosity` | Total luminositas piksel dalam area cacat |
| 9 | `Minimum_of_Luminosity` | Nilai luminositas minimum |
| 10 | `Maximum_of_Luminosity` | Nilai luminositas maksimum |
| 11 | `Length_of_Conveyer` | Panjang conveyor tempat baja berada |
| 12 | `TypeOfSteel_A300` | Indikator tipe baja A300 (0 atau 1) |
| 13 | `TypeOfSteel_A400` | Indikator tipe baja A400 (0 atau 1) |
| 14 | `Steel_Plate_Thickness` | Ketebalan pelat baja |

#### Fitur Kontinu (13 fitur)

| No | Nama Fitur | Deskripsi |
|----|-----------|-----------|
| 15 | `Edges_Index` | Indeks tepi cacat (rasio piksel tepi terhadap total) |
| 16 | `Empty_Index` | Indeks kekosongan area cacat |
| 17 | `Square_Index` | Indeks kotak (mendekati 1 = bentuk persegi) |
| 18 | `Outside_X_Index` | Indeks posisi di luar batas X |
| 19 | `Edges_X_Index` | Indeks tepi pada arah X |
| 20 | `Edges_Y_Index` | Indeks tepi pada arah Y |
| 21 | `Outside_Global_Index` | Indeks posisi global di luar batas |
| 22 | `LogOfAreas` | Logaritma natural dari luas area cacat |
| 23 | `Log_X_Index` | Logaritma dari indeks X |
| 24 | `Log_Y_Index` | Logaritma dari indeks Y |
| 25 | `Orientation_Index` | Indeks orientasi cacat (−1 s.d. 1) |
| 26 | `Luminosity_Index` | Indeks luminositas (−1 s.d. 0) |
| 27 | `SigmoidOfAreas` | Transformasi sigmoid dari luas area |

### Kelas Target (7 kelas)

| Kelas | Deskripsi | Bobot Risiko Fuzzy |
|-------|-----------|-------------------|
| `Pastry` | Cacat jenis pastry (gelembung di permukaan) | 0.75 (tinggi) |
| `Z_Scratch` | Goresan berbentuk Z | 0.88 (sangat tinggi) |
| `K_Scatch` | Goresan berbentuk K | **0.95 (kritis)** |
| `Stains` | Noda pada permukaan | 0.35 (rendah) |
| `Dirtiness` | Kotoran menempel | 0.40 (rendah) |
| `Bumps` | Benjolan/tonjolan | 0.70 (tinggi) |
| `Other_Faults` | Cacat lain yang tidak terklasifikasi | 0.55 (sedang) |

### Cara Membaca Data

```python
# Di dalam load_steel_data() — kelompok_x4_hybrid_pipeline.py
df = pd.read_csv("data/Faults.NNA", sep=r"\s+", header=None, names=all_col_names)
X = df[ALL_FEATURES]                       # 27 kolom fitur
y = df[TARGET_CLASSES].idxmax(axis=1)      # one-hot → nama kelas
y_encoded = LabelEncoder().fit_transform(y) # nama kelas → integer (0-6)
```

---

## 🏗️ Arsitektur Hybrid Pipeline

```
Dataset (1941 × 27)
        │
        ▼
┌──────────────────────────┐
│  1. Preprocessing         │  StandardScaler + LabelEncoder + Split 80/20
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  2. Genetic Algorithm     │  Seleksi Fitur (Binary Chromosome)
│     (GA)                 │  → Subset fitur terpilih
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  3. Particle Swarm Optim │  Tuning Hyperparameter MLP (h1, h2, lr, α)
│     (PSO)                │  → Hyperparameter optimal
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  4. Training 6 Model     │  Model A/B/C/D (MLP) + Model E/F (1D-CNN)
│     Neural Networks      │  → Probabilitas kelas per sampel
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  5. Fuzzy Inference Sys. │  Risk Score + Prioritas Inspeksi
│     (Decision Support)   │  → Accept / Monitor / Rework / Reject
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  6. Streamlit Dashboard  │  Visualisasi & Prediksi Real-Time
│     (7 Halaman)          │
└──────────────────────────┘
```

---

## 📄 Dokumentasi Kode: `kelompok_x4_hybrid_pipeline.py`

### Konstanta Global

```python
RANDOM_STATE = 42            # Seed untuk reproduksibilitas
INTEGER_FEATURES = [...]     # 14 fitur integer
CONTINUOUS_FEATURES = [...]  # 13 fitur kontinu
ALL_FEATURES = INTEGER_FEATURES + CONTINUOUS_FEATURES   # 27 total
TARGET_CLASSES = ["Pastry", "Z_Scratch", "K_Scatch", "Stains",
                  "Dirtiness", "Bumps", "Other_Faults"]
ARTIFACT_PATH = "pipeline_artifacts.pkl"
```

---

### Fungsi `load_steel_data()`

**Lokasi:** Baris ~112–152

**Tujuan:** Memuat dataset dari file lokal dan melakukan encoding target.

**Parameter:**
- `data_dir` *(Optional[str])*: Path ke folder data. Default: `./data/`

**Return:**
- `X` *(pd.DataFrame, shape 1941×27)*: Matriks fitur
- `y` *(pd.Series)*: Label kelas integer (0–6)
- `le` *(LabelEncoder)*: Encoder yang sudah di-fit (untuk `inverse_transform`)

**Alur Kerja:**
1. Baca nama variabel dari `Faults27x7_var`
2. Muat `Faults.NNA` sebagai space-separated CSV tanpa header
3. Ambil 27 kolom fitur → `X`
4. Kolom target one-hot → ambil nama kolom dengan nilai 1 (`idxmax`) → encode ke integer

---

### Class `GeneticFeatureSelector`

**Lokasi:** Baris ~159–306

**Tujuan:** Genetic Algorithm (GA) untuk seleksi subset fitur optimal dari 27 fitur.

#### Parameter Konstruktor

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `n_features` | 27 | Jumlah fitur input |
| `pop_size` | 12 | Ukuran populasi per generasi |
| `generations` | 10 | Jumlah iterasi GA |
| `mutation_rate` | 0.1 | Probabilitas flip bit per gen |
| `min_features` | 3 | Minimum fitur yang harus aktif |
| `random_state` | 42 | Seed RNG |

#### Representasi Kromosom

Setiap individu adalah vektor biner panjang 27:
```
[1, 0, 1, 1, 0, 1, ..., 0]  ← 1 = fitur dipakai, 0 = dibuang
```

#### Method Utama

| Method | Deskripsi |
|--------|-----------|
| `_fitness(chromosome, X_tr, y_tr)` | 3-fold CV F1-macro dengan penalti jumlah fitur |
| `_tournament_select(population, fitness)` | Pilih individu terbaik dari duel 2 kandidat |
| `_crossover(p1, p2)` | Single-point crossover |
| `_mutate(chromosome)` | Bit-flip random + repair minimum fitur |
| `fit(X_tr, y_tr)` | Jalankan GA, simpan kromosom terbaik |
| `selected_feature_names(feature_names)` | Kembalikan nama fitur yang dipilih |

#### Alur GA

```
Inisialisasi populasi (random binary)
└── Untuk setiap generasi:
    ├── Hitung fitness tiap individu (3-fold CV F1-macro)
    ├── Simpan kromosom terbaik (elitism)
    ├── Buat generasi baru:
    │   ├── Salin kromosom terbaik (elitism)
    │   └── Loop: tournament select → crossover → mutasi
    └── Update populasi
```

#### Bug Fix #1 — Tournament Selection

**Bug asli:** `rng.choice` dipanggil dua kali secara independen sehingga individu yang dikembalikan tidak menjamin menang turnamen.  
**Fix:** Dua indeks ditarik sekali (`replace=False`), lalu bandingkan `fitness_scores[idx1]` vs `fitness_scores[idx2]`.

---

### Class `PSOHyperparameterOptimizer`

**Lokasi:** Baris ~313–416

**Tujuan:** Particle Swarm Optimization (PSO) untuk mencari hyperparameter MLP terbaik pada fitur yang sudah diseleksi GA.

#### Ruang Pencarian 4 Dimensi

| Dimensi | Variabel | Rentang |
|---------|----------|---------|
| 0 | `h1` — neuron hidden layer 1 | [32, 160] |
| 1 | `h2` — neuron hidden layer 2 | [16, 128] |
| 2 | `lr` — learning rate MLP | [0.0005, 0.01] |
| 3 | `alpha` — regularisasi L2 | [0.0001, 0.01] |

#### Parameter Konstruktor

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `n_particles` | 8 | Jumlah partikel |
| `iterations` | 8 | Jumlah iterasi PSO |
| `w` | 0.7 | Inertia weight |
| `c1` | 1.5 | Koefisien kognitif (personal best) |
| `c2` | 1.5 | Koefisien sosial (global best) |

#### Update Kecepatan PSO

$$v_i(t+1) = w \cdot v_i(t) + c_1 r_1 (p_{best,i} - x_i) + c_2 r_2 (g_{best} - x_i)$$

#### Method Utama

| Method | Deskripsi |
|--------|-----------|
| `_decode(pos)` | Konversi vektor posisi → dict hyperparameter MLP |
| `_evaluate(pos, X_tr, y_tr)` | 3-fold CV F1-macro untuk satu partikel |
| `fit(X_tr, y_tr)` | Jalankan PSO, simpan posisi global best |
| `best_params` *(property)* | Kembalikan hyperparameter optimal |

---

### Class `SteelInspectionFuzzySystem`

**Lokasi:** Baris ~423–546

**Tujuan:** Sistem inferensi fuzzy untuk mengonversi output model (kelas + confidence + area cacat) menjadi keputusan prioritas inspeksi.

#### Input Fuzzy

| Input | Rentang | Deskripsi |
|-------|---------|-----------|
| `confidence_prob` | [0, 1] | Max softmax probability dari model |
| `normalized_area` | [0, 1] | (LogOfAreas − min) / (max − min) |
| `predicted_class` | string | Nama kelas cacat yang diprediksi |

#### Fungsi Keanggotaan

```
Confidence:
  prob_low  = trapmf(conf, 0.0, 0.0, 0.35, 0.55)   ← "rendah"
  prob_med  = trimf (conf, 0.40, 0.65, 0.85)         ← "sedang"
  prob_high = trapmf(conf, 0.70, 0.85, 1.0,  1.0)   ← "tinggi"

Area:
  area_small = trapmf(area, 0.0, 0.0, 0.25, 0.45)   ← "kecil"
  area_mod   = trimf (area, 0.30, 0.55, 0.75)        ← "sedang"
  area_large = trapmf(area, 0.65, 0.80, 1.0,  1.0)  ← "besar"
```

#### Risk Score & Keputusan

```python
risk = 0.40 * confidence + 0.30 * norm_area + 0.30 * class_severity

# Penyesuaian berbasis aturan:
if r_reject > 0.5:  risk = max(risk, 0.82)   # paksa risiko tinggi
if r_accept > 0.5 and class_risk <= 0.45:
                    risk = min(risk, 0.25)    # paksa risiko rendah

# Defuzzifikasi:
risk < 0.30  → "Accept"
risk < 0.55  → "Monitor"
risk < 0.78  → "Rework"
else         → "Reject"
```

#### Bug Fix #2 — trapmf/trimf

**Bug asli:** Pembagian `(x-a)/(b-a+1e-9)` — guard diterapkan ke seluruh ekspresi, bukan hanya penyebut.  
**Fix:** `max(denominator, eps)` diterapkan tepat di penyebut saja, dengan boundary check eksplisit sebelum masuk ke ramp computation.

---

### Class `CNN1DNet` (PyTorch)

**Lokasi:** Baris ~553–579

**Tujuan:** Arsitektur jaringan saraf 1D-CNN berbasis PyTorch untuk klasifikasi data tabular baja.

#### Arsitektur

```
Input (batch, 1, n_features)
       ↓
Conv1d(1→32, kernel=3, padding=1)  → BatchNorm1d(32) → ReLU
       ↓
Conv1d(32→64, kernel=3, padding=1) → BatchNorm1d(64) → ReLU
       ↓
AdaptiveAvgPool1d(4)   → output: (batch, 64, 4)
       ↓
Flatten                → (batch, 256)
       ↓
Dropout(0.3) → Linear(256→64) → ReLU → Dropout(0.2) → Linear(64→7)
       ↓
Logits (batch, 7)
```

**Catatan:** Input 2D `(batch, features)` secara otomatis di-`unsqueeze(1)` menjadi `(batch, 1, features)` di dalam `forward()`.

---

### Class `CNN1DClassifier`

**Lokasi:** Baris ~582–641

**Tujuan:** Wrapper sklearn-compatible untuk `CNN1DNet` agar bisa dipakai seperti `MLPClassifier`.

#### Parameter Konstruktor

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `epochs` | 25 | Jumlah epoch training |
| `batch_size` | 32 | Ukuran mini-batch |
| `lr` | 0.003 | Learning rate Adam |
| `random_state` | 42 | Seed |

#### Method

| Method | Deskripsi |
|--------|-----------|
| `fit(X, y)` | Train model dengan Adam + CrossEntropyLoss |
| `predict(X)` | Prediksi kelas (argmax dari softmax) |
| `predict_proba(X)` | Probabilitas semua kelas (softmax) |

#### Catatan Penting — Pickle Compatibility

```python
# Wajib ada agar joblib.load() berhasil di Streamlit:
CNN1DNet.__module__ = "kelompok_x4_hybrid_pipeline"
CNN1DClassifier.__module__ = "kelompok_x4_hybrid_pipeline"
```

Tanpa ini, saat pipeline dijalankan sebagai `__main__`, modul akan disimpan sebagai `__main__.CNN1DClassifier` dan gagal di-load oleh Streamlit.

---

### Fungsi `evaluate_classifier()`

**Lokasi:** Baris ~648–669

**Tujuan:** Menghitung 5 metrik evaluasi untuk satu model.

**Return:** Dict berisi `Accuracy`, `Precision (Macro)`, `Recall (Macro)`, `F1-Score (Macro)`, `F1-Score (Weighted)` — semua dibulatkan 4 desimal.

---

### Fungsi `run_fuzzy_on_test()`

**Lokasi:** Baris ~672–726

**Tujuan:** Menjalankan layer Fuzzy pada N sampel pertama test set dan mengembalikan tabel hasil.

**Return:** `pd.DataFrame` dengan kolom:
- `Sample_ID`, `True_Class`, `Predicted_Class`
- `Confidence`, `LogOfAreas_Raw`, `LogOfAreas_Norm`
- `Fuzzy_Risk_Score`, `Inspection_Priority`

---

### Fungsi `run_full_pipeline()`

**Lokasi:** Baris ~733–akhir

**Tujuan:** Orkestrasi seluruh pipeline dari awal hingga simpan artefak.

#### Parameter

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `ga_runs` | 3 | Berapa kali GA dijalankan (ambil kromosom terbaik) |
| `ga_pop` | 12 | Ukuran populasi GA |
| `ga_gen` | 10 | Jumlah generasi GA |
| `pso_particles` | 8 | Jumlah partikel PSO |
| `pso_iter` | 8 | Jumlah iterasi PSO |
| `data_dir` | None | Path folder data |
| `artifact_path` | `pipeline_artifacts.pkl` | Path output pkl |

#### Alur Langkah

```
1. load_steel_data()              → X, y, le
2. StandardScaler.fit_transform() → X_train_sc, X_test_sc
3. GA × ga_runs runs              → best_chromosome, ga_selected_features
4. PSO (4-D) on GA features       → pso_params (h1, h2, lr, alpha)
5. Train 6 model:
   A: MLPClassifier(64,32) semua fitur
   B: MLPClassifier(64,32) GA fitur
   C: MLPClassifier(h1,h2) semua fitur (PSO tuned)
   D: MLPClassifier(h1,h2) GA fitur  (PSO tuned) ← Full Hybrid MLP
   E: CNN1DClassifier() semua fitur
   F: CNN1DClassifier() GA fitur                 ← Full Hybrid CNN
6. evaluate_classifier() semua model → comparison_df
7. run_fuzzy_on_test(Model D) → fuzzy_results_df
8. joblib.dump(artifacts, pkl)
```

---

## 📊 6 Model yang Dibandingkan

| Model | Nama | Fitur | Arsitektur |
|-------|------|-------|-----------|
| **A** | Baseline MLP | Semua 27 | MLP (64, 32) default |
| **B** | GA + MLP | GA selected | MLP (64, 32) default |
| **C** | PSO + MLP | Semua 27 | MLP (h1, h2) PSO-tuned |
| **D** | GA + PSO + MLP | GA selected | MLP (h1, h2) PSO-tuned ✅ |
| **E** | 1D-CNN All Feat | Semua 27 | 1D-CNN PyTorch |
| **F** | GA + 1D-CNN | GA selected | 1D-CNN PyTorch 🚀 |

---

## 🗃️ File Artefak: `pipeline_artifacts.pkl`

File ini dibuat oleh `run_full_pipeline()` dan dimuat oleh Streamlit dashboard.

| Key | Tipe | Deskripsi |
|-----|------|-----------|
| `scaler` | `StandardScaler` | Scaler yang sudah di-fit pada training data |
| `le` | `LabelEncoder` | Encoder kelas (0–6 ↔ nama kelas) |
| `all_feature_names` | `List[str]` | 27 nama fitur |
| `class_names` | `List[str]` | 7 nama kelas target |
| `ga_chromosome` | `np.ndarray (27,)` | Kromosom biner terbaik GA |
| `ga_selected_features` | `List[str]` | Nama fitur yang dipilih GA |
| `ga_feature_freq` | `Dict` | Frekuensi seleksi tiap fitur |
| `ga_curve` | `List[float]` | Kurva konvergensi GA |
| `ga_all_results` | `List[Dict]` | Hasil semua GA runs |
| `pso_params` | `Dict` | Hyperparameter terbaik PSO {h1, h2, lr, alpha} |
| `pso_curve` | `List[float]` | Kurva konvergensi PSO |
| `pso_best_fitness` | `float` | Fitness PSO terbaik |
| `final_model` | `MLPClassifier` | Model D (GA+PSO+MLP) yang sudah di-train |
| `final_model_cnn` | `CNN1DClassifier` | Model F (GA+1D-CNN) yang sudah di-train |
| `all_models` | `Dict[str, model]` | Semua 6 model |
| `comparison_df` | `pd.DataFrame` | Tabel metrik 6 model |
| `confusion_matrix` | `np.ndarray` | CM Model D |
| `classification_report` | `str` | Laporan klasifikasi Model D |
| `confusion_matrix_cnn` | `np.ndarray` | CM Model F |
| `classification_report_cnn` | `str` | Laporan klasifikasi Model F |
| `X_test_sc` | `np.ndarray` | Test set (scaled, semua fitur) |
| `X_test_ga` | `np.ndarray` | Test set (scaled, GA fitur saja) |
| `X_test_orig` | `pd.DataFrame` | Test set asli (sebelum scaling) |
| `y_test` | `np.ndarray` | Label test set |
| `fuzzy_results_df` | `pd.DataFrame` | Hasil fuzzy 200 sampel test |

---

## 📺 Dokumentasi Dashboard: `streamlit_dashboard.py`

Dashboard dijalankan dengan:
```bash
streamlit run streamlit_dashboard.py
```

### Halaman 1 — 🏠 Beranda

**Tujuan:** Pengenalan proyek dan arsitektur pipeline.

**Konten:**
- Header proyek (nama, mata kuliah, institusi)
- Diagram alur pipeline hybrid AI (5 langkah)
- Deskripsi singkat setiap komponen

---

### Halaman 2 — 📊 Eksplorasi Data

**Tujuan:** Analisis visual dataset sebelum training.

**Tab yang tersedia:**
1. **Distribusi Kelas** — Bar chart jumlah sampel per kelas cacat
2. **Korelasi Fitur** — Heatmap korelasi 27 fitur
3. **Box Plot** — Distribusi fitur per kelas (dropdown pilih fitur)
4. **Statistik Deskriptif** — Tabel mean, std, min, max, quartile

---

### Halaman 3 — 🧬 Genetic Algorithm

**Tujuan:** Visualisasi proses dan hasil seleksi fitur GA.

**Tab yang tersedia:**
1. **Convergence Curves** — Plot fitness GA per generasi (semua runs)
2. **Fitur Terpilih** — Bar chart frekuensi fitur terpilih + highlight final
3. **Statistik Runs** — Tabel perbandingan tiap GA run

**Metrik yang ditampilkan:**
- Jumlah fitur asli (27) vs terpilih
- Persentase reduksi dimensi
- Jumlah GA runs

---

### Halaman 4 — 🔵 Particle Swarm Optimization

**Tujuan:** Visualisasi proses dan hasil tuning hyperparameter PSO.

**Konten:**
- Kurva konvergensi PSO
- Tabel hyperparameter optimal: `h1`, `h2`, `lr`, `alpha`
- Interpretasi hasil PSO

---

### Halaman 5 — 🤖 Perbandingan Model

**Tujuan:** Membandingkan performa 6 model secara kuantitatif.

**Tab yang tersedia:**

**Tab 1 — Tabel & Bar Chart:**
- Tabel metrik 6 model (highlight sel terbaik dengan warna hijau)
- Bar chart F1-Macro ke-6 model dengan label nilai

**Tab 2 — Confusion Matrix:**
- Radio button: pilih Model D atau Model F
- Heatmap confusion matrix dengan annotation

**Tab 3 — Classification Report:**
- Tab nested: Model D dan Model F
- Laporan precision/recall/f1 per kelas dalam format teks

---

### Halaman 6 — 🔶 Sistem Fuzzy

**Tujuan:** Visualisasi dan analisis hasil sistem inferensi fuzzy.

**Konten:**
- Distribusi keputusan (Accept/Monitor/Rework/Reject) — pie chart
- Scatter plot risk score vs confidence (warna per keputusan)
- Tabel 200 sampel hasil fuzzy (dengan color gradient pada risk score)
- Statistik ringkas per kelas keputusan

---

### Halaman 7 — 🔮 Prediksi Real-Time

**Tujuan:** Input nilai fitur manual dan dapatkan prediksi + keputusan fuzzy instan.

**Alur:**
1. Pilih model (MLP → Model D, atau 1D-CNN → Model F)
2. Input 27 nilai fitur via slider (dengan nilai default representatif)
3. Klik tombol **"Prediksi Cacat"**
4. Output:
   - Kelas cacat terprediksi
   - Probabilitas semua 7 kelas (bar chart)
   - Risk score fuzzy + keputusan inspeksi
   - Tabel lengkap probabilitas

---

## 🐛 Daftar Bug Fix

| No | Bug | Lokasi | Fix |
|----|-----|--------|-----|
| **FIX-1** | Tournament selection mengindeks `fitness_scores` dengan indeks random baru | `_tournament_select()` | Dua indeks ditarik sekaligus; bandingkan `fitness_scores[idx1]` vs `fitness_scores[idx2]` |
| **FIX-2** | `trapmf`/`trimf` guard epsilon di ekspresi salah | `trapmf()`, `trimf()` | `max(denominator, eps)` hanya di penyebut; boundary check eksplisit |
| **FIX-3** | Import `os`, `joblib` tidak ada | Baris 44–68 | Ditambahkan |
| **FIX-4** | Model tidak disimpan | `run_full_pipeline()` | `joblib.dump(artifacts, pkl)` ditambahkan |
| **FIX-5** | Block `__main__` kosong | Baris ~1000+ | Argparse CLI + demo prediksi |
| **FIX-6** | `X_test_orig` tidak disimpan sebelum scaling | Preprocessing | `X_test_orig = X_test.copy()` sebelum `transform()` |
| **FIX-7** | Dataset dimuat via UCI online (rawan gagal) | `load_steel_data()` | Ganti ke pembacaan lokal `Faults.NNA` + `Faults27x7_var` |
| **FIX-8** | 27 nama fitur salah/tidak lengkap | Konstanta | Diambil dari `Faults27x7_var` yang benar |
| **FIX-9** | Dimensi PSO tidak konsisten antara .py dan notebook | `PSOHyperparameterOptimizer` | Gunakan 4-D (h1, h2, lr, alpha) secara konsisten |
| **FIX-10** | `ax.boxplot(..., labels=...)` deprecated di matplotlib ≥3.9 | `streamlit_dashboard.py` baris ~395 | `tick_labels=` dengan try/except fallback |
| **FIX-11** | `BrokenProcessPool: bad allocation` di Windows (PyTorch + joblib) | GA fitness, PSO evaluate | `n_jobs=1` di semua `cross_val_score` |
| **FIX-12** | `AttributeError: '__main__' has no attribute 'CNN1DClassifier'` saat load pkl | `streamlit_dashboard.py` | Set `__module__` ke `"kelompok_x4_hybrid_pipeline"` + registrasi `sys.modules` |

---

## ⚙️ Cara Menjalankan

### 1. Persiapan Environment

```bash
# Install dependensi
pip install streamlit scikit-learn pandas numpy matplotlib seaborn joblib
pip install torch  # CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Training Pipeline (jika pkl belum ada)

```bash
cd kelompok_x4_steel_fault

# Mode cepat (untuk testing):
python kelompok_x4_hybrid_pipeline.py --ga-runs 2 --ga-pop 8 --ga-gen 6 --pso-particles 6 --pso-iter 6

# Mode penuh:
python kelompok_x4_hybrid_pipeline.py --ga-runs 3 --ga-pop 12 --ga-gen 10 --pso-particles 8 --pso-iter 8
```

### 3. Jalankan Dashboard Streamlit

```bash
streamlit run streamlit_dashboard.py
# Buka browser: http://localhost:8501
```

### 4. Verifikasi Model di pkl

```python
import sys, joblib
from kelompok_x4_hybrid_pipeline import CNN1DClassifier, CNN1DNet
sys.modules["__main__"].CNN1DClassifier = CNN1DClassifier
sys.modules["__main__"].CNN1DNet = CNN1DNet

arts = joblib.load("pipeline_artifacts.pkl")
print(list(arts["comparison_df"].index))
# Output: ['Model A: Baseline MLP', 'Model B: GA + MLP', ..., 'Model F: GA + 1D-CNN']
```

---

## 📦 Dependensi

| Library | Versi Teruji | Fungsi |
|---------|-------------|--------|
| Python | 3.14.7 | Runtime |
| PyTorch (`torch`) | 2.13.0+cpu | 1D-CNN |
| scikit-learn | 1.9.1 | MLP, preprocessing, metrics |
| pandas | — | Manipulasi data |
| numpy | — | Komputasi numerik |
| matplotlib | 3.11.2 | Visualisasi |
| seaborn | — | Heatmap confusion matrix |
| joblib | — | Serialisasi model |
| streamlit | 1.67.0 | Dashboard web |

---

*Dokumentasi ini dibuat otomatis dan sesuai dengan kode pada commit terakhir.*  
*Kelompok X4 | TE2514031 | Institut Teknologi Kalimantan | 2026/2027*
