# 📄 Dokumentasi: 1D-CNN (1D Convolutional Neural Network) — PyTorch Classifier
**Kelompok X4 | TE2514031 | Institut Teknologi Kalimantan**

---

## 1. Pengertian

**1D Convolutional Neural Network (1D-CNN)** menggunakan operasi konvolusi 1D untuk mengekstrak pola lokal dari urutan data. Meskipun biasanya dipakai untuk data sekuensial (sinyal, teks), 1D-CNN juga efektif untuk data tabular karena mampu mendeteksi pola antar-fitur yang berdekatan.

Dalam proyek ini, 1D-CNN digunakan sebagai alternatif MLP dengan **2 varian model**:

| Model | Nama | Fitur |
|-------|------|-------|
| **E** | 1D-CNN All Feat | Semua 27 fitur |
| **F** | GA + 1D-CNN | GA-selected features |

---

## 2. Arsitektur CNN1DNet

```
Input: (batch, n_features)
         │
         │ unsqueeze(1)
         ↓
(batch, 1, n_features)
         │
┌────────────────────────────────┐
│  Conv1d(1→32, kernel=3, pad=1) │
│  BatchNorm1d(32)               │
│  ReLU                          │
└────────────────────────────────┘
         │
(batch, 32, n_features)
         │
┌────────────────────────────────┐
│  Conv1d(32→64, kernel=3, pad=1)│
│  BatchNorm1d(64)               │
│  ReLU                          │
└────────────────────────────────┘
         │
(batch, 64, n_features)
         │
┌────────────────────────────────┐
│  AdaptiveAvgPool1d(4)          │
└────────────────────────────────┘
         │
(batch, 64, 4)
         │  Flatten
         ↓
(batch, 256)
         │
┌────────────────────────────────┐
│  Dropout(0.3)                  │
│  Linear(256 → 64)              │
│  ReLU                          │
│  Dropout(0.2)                  │
│  Linear(64 → n_classes)        │
└────────────────────────────────┘
         │
(batch, 7)   ← logits
         │  Softmax
         ↓
Probabilitas kelas [0,1]^7
```

---

## 3. Formula Matematis

### 3.1 Operasi Konvolusi 1D

Untuk input $\mathbf{x}$ dengan $C_{in}$ channel dan filter $\mathbf{W}_k$ dengan kernel size $K$:

$$(\mathbf{x} * \mathbf{W}_k)[t] = \sum_{c=1}^{C_{in}} \sum_{\tau=0}^{K-1} x_c[t + \tau] \cdot W_{k,c}[\tau]$$

| Simbol | Deskripsi |
|--------|-----------|
| $t$ | Posisi output |
| $K$ | Ukuran kernel = **3** |
| $C_{in}$ | Jumlah input channel |
| $W_{k,c}$ | Filter ke-$k$ pada channel $c$ |

**Padding same** (`padding=1`): output size = input size (tidak ada shrinkage).

### 3.2 Batch Normalization

Setelah setiap Conv1d, output dinormalisasi per batch:

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y_i = \gamma \hat{x}_i + \beta$$

| Simbol | Deskripsi |
|--------|-----------|
| $\mu_B$ | Mean mini-batch |
| $\sigma_B^2$ | Variance mini-batch |
| $\epsilon$ | Konstanta kecil (1e-5) untuk stabilitas |
| $\gamma, \beta$ | Parameter yang dipelajari (scale & shift) |

**Manfaat:** Mempercepat konvergensi, mengurangi sensitivitas terhadap learning rate, menggantikan sebagian fungsi dropout.

### 3.3 ReLU Activation

$$\text{ReLU}(z) = \max(0, z)$$

### 3.4 Adaptive Average Pooling

Mereduksi dimensi temporal ke ukuran tetap **4**:

$$\text{pool}[k] = \frac{1}{|S_k|} \sum_{t \in S_k} x[t]$$

Di mana $S_k$ adalah subset indeks untuk output ke-$k$ (dihitung otomatis).

**Output shape:** `(batch, 64, 4)` → setelah flatten: `(batch, 256)`

### 3.5 Dropout

Regularisasi dengan menonaktifkan neuron secara acak saat training:

$$\hat{x}_i = \begin{cases} \frac{x_i}{1-p} & \text{dengan probabilitas } 1-p \\ 0 & \text{dengan probabilitas } p \end{cases}$$

| Layer | Dropout Rate |
|-------|-------------|
| Sebelum Linear 1 | $p = 0.3$ |
| Sebelum Linear 2 | $p = 0.2$ |

### 3.6 Linear (Fully Connected) Layer

$$\mathbf{z} = \mathbf{W} \mathbf{a} + \mathbf{b}$$

Layer 1: $256 \rightarrow 64$ | Layer 2: $64 \rightarrow 7$

### 3.7 Loss Function — Cross-Entropy

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log \hat{y}_{i, y_i}$$

Di mana $\hat{y}_{i, y_i}$ adalah probabilitas softmax untuk kelas benar sampel $i$.

### 3.8 Optimizer — Adam

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla_\theta \mathcal{L}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla_\theta \mathcal{L})^2$$
$$\theta_{t+1} = \theta_t - \frac{lr}{\sqrt{\hat{v}_t} + \epsilon}\hat{m}_t$$

| Parameter | Nilai |
|-----------|-------|
| $lr$ | **0.003** |
| $\beta_1$ | 0.9 |
| $\beta_2$ | 0.999 |
| `weight_decay` | **1e-4** (L2 regularization) |

---

## 4. Jumlah Parameter Model

Untuk input 27 fitur (Model E — semua fitur):

| Layer | Shape | Parameter |
|-------|-------|-----------|
| Conv1d (1→32, k=3) | 1×3×32 + 32 | **128** |
| BN1 | 2×32 | **64** |
| Conv1d (32→64, k=3) | 32×3×64 + 64 | **6,208** |
| BN2 | 2×64 | **128** |
| Linear (256→64) | 256×64 + 64 | **16,448** |
| Linear (64→7) | 64×7 + 7 | **455** |
| **Total** | | **~23,431 params** |

---

## 5. Training Loop

```python
# 25 epoch, batch_size = 32, Adam lr=0.003
model.train()
for epoch in range(25):
    for batch_x, batch_y in DataLoader(dataset, batch_size=32, shuffle=True):
        optimizer.zero_grad()
        logits = model(batch_x)              # forward pass
        loss = CrossEntropyLoss(logits, batch_y)
        loss.backward()                      # backpropagation
        optimizer.step()                     # update weights
```

---

## 6. Pickle Compatibility Fix

Karena pipeline dijalankan sebagai `__main__`, modul perlu diset ulang agar `joblib.load()` di Streamlit bisa mengenali kelasnya:

```python
CNN1DNet.__module__        = "kelompok_x4_hybrid_pipeline"
CNN1DClassifier.__module__ = "kelompok_x4_hybrid_pipeline"
```

Dan di `streamlit_dashboard.py`:
```python
import sys
from kelompok_x4_hybrid_pipeline import CNN1DClassifier, CNN1DNet
sys.modules["__main__"].CNN1DClassifier = CNN1DClassifier
sys.modules["__main__"].CNN1DNet = CNN1DNet
```

---

## 7. Cara Menggunakan Modul

```python
from modules.cnn1d_classifier import CNN1DClassifier, CNN1DNet

# Model E: 1D-CNN semua 27 fitur
cnn_e = CNN1DClassifier(epochs=25, batch_size=32, lr=0.003, random_state=42)
cnn_e.fit(X_train_sc, y_train)       # X_train_sc: semua 27 fitur

# Model F: 1D-CNN fitur GA-selected
cnn_f = CNN1DClassifier(epochs=25, batch_size=32, lr=0.003, random_state=42)
cnn_f.fit(X_train_ga, y_train)       # X_train_ga: GA fitur

# Prediksi
y_pred = cnn_f.predict(X_test_ga)
proba  = cnn_f.predict_proba(X_test_ga)   # shape (n, 7)
```

---

## 8. Cara Membuka Hasil dari pipeline_artifacts.pkl

```python
import sys, joblib
from kelompok_x4_hybrid_pipeline import CNN1DClassifier, CNN1DNet
sys.modules["__main__"].CNN1DClassifier = CNN1DClassifier
sys.modules["__main__"].CNN1DNet = CNN1DNet

arts = joblib.load("pipeline_artifacts.pkl")

# Model F (GA + 1D-CNN)
model_f = arts["final_model_cnn"]             # CNN1DClassifier sudah di-fit
print(type(model_f))                          # CNN1DClassifier
print(model_f.model_)                         # CNN1DNet (PyTorch module)

# Semua model
all_models = arts["all_models"]
model_e = all_models["Model E: 1D-CNN (All Feat)"]
model_f = all_models["Model F: GA + 1D-CNN"]

# Confusion matrix dan classification report Model F
cm_f = arts["confusion_matrix_cnn"]
cr_f = arts["classification_report_cnn"]

# Prediksi baru dengan Model F
y_pred = model_f.predict(arts["X_test_ga"])
proba  = model_f.predict_proba(arts["X_test_ga"])

# Inferensi satu sampel baru
import numpy as np
x_new = np.array([[...]])     # shape (1, n_ga_features)
x_new_sc = arts["scaler"].transform(x_new)[:, arts["ga_chromosome"] == 1]
pred_class = arts["le"].inverse_transform(model_f.predict(x_new_sc))
```

---

## 9. Referensi

- LeCun, Y., Boser, B., Denker, J., et al. (1989). Backpropagation applied to handwritten zip code recognition. *Neural Computation*, 1(4), 541–551.
- Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *ICML 2015*.
- Srivastava, N., et al. (2014). Dropout: A simple way to prevent neural networks from overfitting. *JMLR*, 15(1), 1929–1958.
- Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *NeurIPS 2019*.

---
*Kelompok X4 | TE2514031 | Institut Teknologi Kalimantan | 2026/2027*
