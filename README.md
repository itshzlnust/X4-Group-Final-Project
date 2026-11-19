# Kelompok X4 – Intelligent Steel Surface Fault Detection

**Mata Kuliah**: Kecerdasan Buatan (TE2514031) – Institut Teknologi Kalimantan  
**Dosen Pengampu**: Happy Aprillia, S.ST., M.T., M.Eng., Ph.D.

**Anggota:**
1. 11231084 – Rahmahdaniah Fitri  
2. 17231008 – Aulia Wahdah Atika Rahmah  
3. 10231042 – Ilham Ahmad Fahriji  

---

## Struktur Proyek

```
kelompok_x4_steel_fault/
├── data/
│   ├── Faults.NNA           ← dataset utama (1941 × 34, tab-separated)
│   └── Faults27x7_var       ← nama 34 kolom (27 fitur + 7 target)
├── kelompok_x4_hybrid_pipeline.py    ← pipeline module (FIXED)
├── Kelompok_X4_Steel_Fault_Hybrid_AI_FIXED.ipynb  ← notebook (FIXED)
├── streamlit_dashboard.py            ← dashboard multi-halaman
├── pipeline_artifacts.pkl            ← hasil pipeline (di-generate saat run)
└── README.md
```

---

## Cara Menjalankan

### 1. Install Dependensi
```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib streamlit
```

### 2. Jalankan Pipeline (Training)
```bash
cd kelompok_x4_steel_fault
python kelompok_x4_hybrid_pipeline.py
```
Atau gunakan argumen custom:
```bash
python kelompok_x4_hybrid_pipeline.py --ga-runs 3 --ga-pop 12 --ga-gen 10 --pso-particles 8 --pso-iter 8
```
Ini akan menghasilkan `pipeline_artifacts.pkl`.

### 3. Jalankan Dashboard Streamlit
```bash
streamlit run streamlit_dashboard.py
```

### 4. Atau gunakan Notebook (alternatif pipeline)
Buka `Kelompok_X4_Steel_Fault_Hybrid_AI_FIXED.ipynb` di Jupyter/VS Code, jalankan semua cell dari atas.  
Cell 13 (terakhir) akan menyimpan `pipeline_artifacts.pkl`.

---

## Arsitektur Pipeline

```
Dataset UCI (Faults.NNA)
        ↓ load_steel_data()
Preprocessing (StandardScaler, LabelEncoder, stratified split 80/20)
        ↓
Genetic Algorithm (GeneticFeatureSelector)
  • 3 independent runs
  • Binary chromosome (27 bit)
  • Fitness: 3-fold CV F1-macro − feature-count penalty
  • Tournament selection (FIXED), elitism, crossover, bit-flip mutation
        ↓ best chromosome (subset fitur)
Particle Swarm Optimization (PSOHyperparameterOptimizer)
  • 4-D search: (h1, h2, learning_rate, alpha)
  • Fitness: 3-fold CV F1-macro
        ↓ optimal (h1, h2, lr, α)
Neural Networks – 6 varian (MLP & 1D-CNN):
  A: Baseline MLP (all features, default params)
  B: GA + MLP (GA features, default params)
  C: PSO + MLP (all features, PSO params)
  D: GA + PSO + MLP (GA features, PSO params)  ← BEST MLP (Full Hybrid)
  E: 1D-CNN PyTorch (all features, Conv1d -> BatchNorm -> ReLU -> Pool -> Linear)
  F: GA + 1D-CNN PyTorch (GA features, Hybrid 1D-CNN)
        ↓ probabilitas kelas (MLP / 1D-CNN)
Fuzzy Inference System (SteelInspectionFuzzySystem)
  • Input: confidence, normalized LogOfAreas, class severity weight
  • Output: Accept / Monitor / Rework / Reject
        ↓
pipeline_artifacts.pkl
        ↓
Streamlit Dashboard (7 halaman)
```

---

## Bug Fixes

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | pipeline.py L139-140 | GA tournament selection: `rng.choice` dipanggil dua kali, indeks kedua tidak berhubungan dengan individu di populasi | Gunakan satu `rng.choice(pop_size, 2, replace=False)`, bandingkan fitness dua indeks |
| 2 | pipeline.py `trapmf` | `+1e-9` pada ekspresi numerator, bukan denominator; memberi hasil tidak terduga saat `a==b` | `max(denominator, eps)` yang bersih; cek batas `x<=a` / `x>=d` terlebih dahulu |
| 3 | notebook Cell 5 | `X_test.iloc[i]['LogOfAreas']` – X_test sudah discale, nilai LogOfAreas tidak valid | Simpan `X_test_orig = X_test.copy()` sebelum `scaler.transform()` |
| 4 | notebook Cell 4 | `display()` hanya berfungsi di Jupyter | Ganti dengan `print()` / `df.to_string()` + try/except import IPython |
| 5 | notebook Cell 3 | `X.columns[best_overall_chrom == 1]` bisa gagal jika panjang tidak cocok | Pakai `np.where(...)[0]` lalu index ke `X.columns` |
| 6 | kedua file | Tidak ada fitur save/load model | `run_full_pipeline()` menyimpan `pipeline_artifacts.pkl` via `joblib` |
| 7 | kedua file | Dataset dari `ucimlrepo` (online fetch) | Diganti dengan pembacaan lokal `Faults.NNA` + `Faults27x7_var` |
| 8 | pipeline.py | 27 nama fitur hardcoded salah (tidak ada `Edges_Y_Index`, `Log_X_Index`, dll.) | Nama fitur diambil dari `Faults27x7_var` yang sebenarnya |
| 9 | pipeline.py | `__main__` block kosong | Diganti dengan argparse CLI lengkap |
