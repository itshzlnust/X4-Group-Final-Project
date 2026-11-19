# 📄 Cara Membuka pipeline_artifacts.pkl
**Kelompok X4 | TE2514031 | Institut Teknologi Kalimantan**

---

## Apa itu pipeline_artifacts.pkl?

File `pipeline_artifacts.pkl` adalah file biner yang menyimpan **semua hasil training pipeline** menggunakan format `joblib`. Berisi 6 model terlatih, scaler, encoder, data test, dan hasil evaluasi.

> **Ukuran:** ~1.7 MB | **Format:** joblib pickle

---

## Cara Membuka (Load)

### Method 1: Python Script (Direkomendasikan)

```python
import sys
import joblib

# WAJIB: registrasi kelas CNN sebelum load, agar pickle bisa mengenali kelasnya
from kelompok_x4_hybrid_pipeline import CNN1DClassifier, CNN1DNet
sys.modules["__main__"].CNN1DClassifier = CNN1DClassifier
sys.modules["__main__"].CNN1DNet = CNN1DNet

# Load file
arts = joblib.load("pipeline_artifacts.pkl")

# Lihat semua key yang tersedia
print(list(arts.keys()))
```

### Method 2: Dari Notebook Jupyter

```python
import sys, joblib
from kelompok_x4_hybrid_pipeline import CNN1DClassifier, CNN1DNet
sys.modules["__main__"].CNN1DClassifier = CNN1DClassifier
sys.modules["__main__"].CNN1DNet = CNN1DNet

arts = joblib.load("pipeline_artifacts.pkl")
arts["comparison_df"]   # tampil sebagai tabel di notebook
```

---

## Daftar Lengkap Key dan Cara Aksesnya

### Preprocessing
```python
scaler      = arts["scaler"]           # StandardScaler (sudah di-fit)
le          = arts["le"]               # LabelEncoder (0-6 <-> nama kelas)
feat_names  = arts["all_feature_names"] # list 27 nama fitur
class_names = arts["class_names"]      # list 7 nama kelas
```

### Genetic Algorithm
```python
chromosome  = arts["ga_chromosome"]       # np.array binary (27,) — fitur terpilih
sel_feat    = arts["ga_selected_features"] # list nama fitur terpilih
feat_freq   = arts["ga_feature_freq"]     # dict {nama_fitur: frekuensi_seleksi}
ga_curve    = arts["ga_curve"]            # list float — konvergensi per generasi
all_runs    = arts["ga_all_results"]      # list dict — hasil semua GA runs
```

### PSO
```python
pso_params  = arts["pso_params"]         # dict {h1, h2, lr, alpha}
pso_curve   = arts["pso_curve"]          # list float — konvergensi per iterasi
pso_fitness = arts["pso_best_fitness"]   # float — fitness PSO terbaik
```

### Model Terlatih
```python
model_D = arts["final_model"]            # MLPClassifier (GA + PSO + MLP)
model_F = arts["final_model_cnn"]        # CNN1DClassifier (GA + 1D-CNN)

# Semua 6 model sekaligus
all_models = arts["all_models"]
# Key: 'Model A: Baseline MLP', 'Model B: GA + MLP', 'Model C: PSO + MLP',
#      'Model D: GA + PSO + MLP', 'Model E: 1D-CNN (All Feat)', 'Model F: GA + 1D-CNN'
```

### Evaluasi & Metrik
```python
comparison_df = arts["comparison_df"]    # pd.DataFrame — metrik semua model

# Confusion matrix
cm_d = arts["confusion_matrix"]          # np.ndarray (7,7) — Model D
cm_f = arts["confusion_matrix_cnn"]      # np.ndarray (7,7) — Model F

# Classification report
cr_d = arts["classification_report"]    # str — Model D
cr_f = arts["classification_report_cnn"] # str — Model F
```

### Data Test
```python
X_test_sc   = arts["X_test_sc"]   # np.ndarray — scaled, semua 27 fitur
X_test_ga   = arts["X_test_ga"]   # np.ndarray — scaled, GA fitur saja
X_test_orig = arts["X_test_orig"] # pd.DataFrame — asli sebelum scaling
y_test      = arts["y_test"]      # np.ndarray — label integer 0-6
```

### Hasil Fuzzy
```python
fuzzy_df = arts["fuzzy_results_df"]     # pd.DataFrame — 200 sampel test
# Kolom: Sample_ID, True_Class, Predicted_Class, Confidence,
#        LogOfAreas_Raw, LogOfAreas_Norm, Fuzzy_Risk_Score, Inspection_Priority
```

---

## Contoh: Prediksi Data Baru

```python
import numpy as np

# Siapkan input (nilai 27 fitur dalam urutan ALL_FEATURES)
x_baru = np.array([[
    50, 100, 270000, 271000,    # X_Min, X_Max, Y_Min, Y_Max
    500, 50, 50,                 # Pixels_Areas, X_Perim, Y_Perim
    100000, 76, 150,             # Sum/Min/Max Luminosity
    1200, 1, 0, 80,             # Length_Conveyer, TypeA300, TypeA400, Thickness
    0.5, 0.35, 0.2, 0.005,     # Edges/Empty/Square/OutsideX Index
    0.47, 0.95, 1.0,            # Edges_X/Y, Outside_Global
    2.43, 0.9, 1.6,             # LogOfAreas, Log_X/Y
    0.82, -0.29, 0.58           # Orientation, Luminosity, Sigmoid
]])

# Scaling
x_sc = arts["scaler"].transform(x_baru)

# Subset fitur GA
x_ga = x_sc[:, arts["ga_chromosome"] == 1]

# Prediksi dengan Model D (MLP)
pred_int = arts["final_model"].predict(x_ga)
pred_cls = arts["le"].inverse_transform(pred_int)
proba    = arts["final_model"].predict_proba(x_ga)

print(f"Kelas terprediksi: {pred_cls[0]}")
print(f"Confidence: {proba[0].max():.4f}")

# Keputusan fuzzy
from modules.fuzzy_system import SteelInspectionFuzzySystem
log_area_norm = 0.6  # contoh normalized area
fuzzy = SteelInspectionFuzzySystem()
decision, risk = fuzzy.evaluate_decision(pred_cls[0], proba[0].max(), log_area_norm)
print(f"Keputusan: {decision} | Risk: {risk:.4f}")
```

---

## Troubleshooting

### Error: `AttributeError: module '__main__' has no attribute 'CNN1DClassifier'`

**Penyebab:** Lupa menambahkan registrasi kelas CNN sebelum `joblib.load()`.

**Fix:**
```python
import sys
from kelompok_x4_hybrid_pipeline import CNN1DClassifier, CNN1DNet
sys.modules["__main__"].CNN1DClassifier = CNN1DClassifier  # wajib
sys.modules["__main__"].CNN1DNet = CNN1DNet                # wajib
# Baru load:
arts = joblib.load("pipeline_artifacts.pkl")
```

### Error: `FileNotFoundError: pipeline_artifacts.pkl`

**Fix:** Pastikan working directory sudah benar:
```python
import os
os.chdir(r"c:\Users\ia679\.gemini\antigravity\scratch\kelompok_x4_steel_fault")
```

---
*Kelompok X4 | TE2514031 | Institut Teknologi Kalimantan | 2026/2027*
