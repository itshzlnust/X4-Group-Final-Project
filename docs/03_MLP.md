# 📄 Dokumentasi: MLP (Multi-Layer Perceptron) — Neural Network Classifier
**Kelompok X4 | TE2514031 | Institut Teknologi Kalimantan**

---

## 1. Pengertian

**Multi-Layer Perceptron (MLP)** adalah jenis jaringan saraf tiruan feedforward dengan satu atau lebih *hidden layer*. MLP belajar memetakan fitur input ke kelas output melalui propagasi maju (*forward pass*) dan pembaruan bobot melalui *backpropagation*.

Dalam proyek ini, MLP digunakan untuk **4 varian model** (A, B, C, D) dengan kombinasi fitur dan hyperparameter berbeda.

---

## 2. Arsitektur MLP

### 2.1 Struktur Umum

```
Input Layer (n_features neuron)
        ↓
Hidden Layer 1 (h1 neuron) + ReLU + Regularisasi L2
        ↓
Hidden Layer 2 (h2 neuron) + ReLU + Regularisasi L2
        ↓
Output Layer (7 neuron = 7 kelas cacat)
        ↓
Softmax → Probabilitas kelas
```

### 2.2 4 Varian Model MLP

| Model | Nama | Fitur | Hidden Layers | Hyperparameter |
|-------|------|-------|---------------|----------------|
| **A** | Baseline MLP | Semua 27 | (64, 32) | default |
| **B** | GA + MLP | GA selected | (64, 32) | default |
| **C** | PSO + MLP | Semua 27 | (h1, h2) PSO-tuned | PSO-tuned |
| **D** | GA + PSO + MLP | GA selected | (h1, h2) PSO-tuned | PSO-tuned |

---

## 3. Formula Matematis

### 3.1 Forward Pass — Satu Layer

Untuk hidden layer $l$:

$$\mathbf{a}^{(l)} = g\left(\mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}\right)$$

| Simbol | Deskripsi |
|--------|-----------|
| $\mathbf{a}^{(l)}$ | Aktivasi output layer $l$ |
| $\mathbf{W}^{(l)}$ | Matriks bobot layer $l$ |
| $\mathbf{b}^{(l)}$ | Vektor bias layer $l$ |
| $g(\cdot)$ | Fungsi aktivasi = **ReLU** |

### 3.2 Fungsi Aktivasi ReLU

$$\text{ReLU}(z) = \max(0, z)$$

### 3.3 Output Layer — Softmax

$$P(y = k \mid \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^{7} e^{z_j}}, \quad k \in \{0, 1, \ldots, 6\}$$

Di mana $\mathbf{z}$ adalah logit dari output layer.

### 3.4 Loss Function — Cross-Entropy

$$\mathcal{L} = -\sum_{i=1}^{N} \sum_{k=1}^{K} y_{ik} \log \hat{y}_{ik}$$

| Simbol | Deskripsi |
|--------|-----------|
| $N$ | Jumlah sampel |
| $K$ | Jumlah kelas = 7 |
| $y_{ik}$ | Label one-hot (1 jika sampel $i$ kelas $k$) |
| $\hat{y}_{ik}$ | Probabilitas prediksi kelas $k$ untuk sampel $i$ |

### 3.5 Regularisasi L2 (Tikhonov)

Bobot dipenalti menggunakan L2 regularization:

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \alpha \sum_{l} \|\mathbf{W}^{(l)}\|_F^2$$

| Simbol | Deskripsi |
|--------|-----------|
| $\alpha$ | Strength regularisasi (dari PSO: optimized) |
| $\|\mathbf{W}\|_F$ | Frobenius norm matriks bobot |

**Efek:** Mencegah overfitting dengan menekan bobot besar.

### 3.6 Backpropagation — Gradient Update (Adam Optimizer)

Adam optimizer (digunakan secara internal oleh sklearn):

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_{t+1} = \theta_t - \frac{lr}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

| Simbol | Nilai Default |
|--------|--------------|
| $\beta_1$ | 0.9 (momentum) |
| $\beta_2$ | 0.999 (RMSProp) |
| $\epsilon$ | 1e-8 |
| $lr$ | Dari PSO untuk Model C/D |

---

## 4. Metrik Evaluasi

### 4.1 Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{\text{prediksi benar}}{\text{total sampel}}$$

### 4.2 Precision (Macro)

$$\text{Precision}_{macro} = \frac{1}{K} \sum_{k=1}^{K} \frac{TP_k}{TP_k + FP_k}$$

### 4.3 Recall (Macro)

$$\text{Recall}_{macro} = \frac{1}{K} \sum_{k=1}^{K} \frac{TP_k}{TP_k + FN_k}$$

### 4.4 F1-Score (Macro)

$$F1_{macro} = \frac{1}{K} \sum_{k=1}^{K} \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}$$

> **Macro averaging:** Setiap kelas diberi bobot yang sama, cocok untuk dataset yang tidak seimbang (imbalanced).

### 4.5 F1-Score (Weighted)

$$F1_{weighted} = \frac{1}{N} \sum_{k=1}^{K} n_k \cdot F1_k$$

Di mana $n_k$ = jumlah sampel kelas $k$.

---

## 5. Konfigurasi sklearn MLPClassifier

```python
from sklearn.neural_network import MLPClassifier

# Model A: Baseline (default hyperparameter)
model_A = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=200,
    random_state=42
)

# Model B: GA features, default hyperparameter
model_B = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=200,
    random_state=42
)

# Model C: PSO hyperparameter, semua fitur
model_C = MLPClassifier(
    hidden_layer_sizes=(h1, h2),          # dari PSO
    learning_rate_init=lr,                # dari PSO
    alpha=alpha,                          # dari PSO
    max_iter=200,
    random_state=42
)

# Model D: GA features + PSO hyperparameter (Full Hybrid MLP)
model_D = MLPClassifier(
    hidden_layer_sizes=(h1, h2),          # dari PSO
    learning_rate_init=lr,                # dari PSO
    alpha=alpha,                          # dari PSO
    max_iter=200,
    random_state=42
)
```

---

## 6. Cara Membuka Hasil dari pipeline_artifacts.pkl

```python
import sys, joblib
from kelompok_x4_hybrid_pipeline import CNN1DClassifier, CNN1DNet
sys.modules["__main__"].CNN1DClassifier = CNN1DClassifier
sys.modules["__main__"].CNN1DNet = CNN1DNet

arts = joblib.load("pipeline_artifacts.pkl")

# Model D (best MLP = GA + PSO + MLP)
model_d = arts["final_model"]           # MLPClassifier sudah di-fit
print(model_d.hidden_layer_sizes)

# Semua model termasuk A, B, C, D
all_models = arts["all_models"]
model_a = all_models["Model A: Baseline MLP"]
model_b = all_models["Model B: GA + MLP"]

# Metrik semua model
comparison_df = arts["comparison_df"]   # pd.DataFrame
print(comparison_df)

# Confusion matrix dan classification report Model D
cm_d = arts["confusion_matrix"]
cr_d = arts["classification_report"]

# Data test
X_test_sc  = arts["X_test_sc"]   # scaled, semua 27 fitur
X_test_ga  = arts["X_test_ga"]   # scaled, GA fitur saja
X_test_orig = arts["X_test_orig"] # asli (sebelum scaling)
y_test     = arts["y_test"]

# Prediksi baru
y_pred = model_d.predict(X_test_ga)
proba  = model_d.predict_proba(X_test_ga)
```

---

## 7. Referensi

- Rumelhart, D.E., Hinton, G.E., & Williams, R.J. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533–536.
- Kingma, D., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015*.
- Hoerl, A.E., & Kennard, R.W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67.

---
*Kelompok X4 | TE2514031 | Institut Teknologi Kalimantan | 2026/2027*
