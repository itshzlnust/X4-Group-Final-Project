# 📄 Dokumentasi: Particle Swarm Optimization (PSO) — Tuning Hyperparameter
**Kelompok X4 | TE2514031 | Institut Teknologi Kalimantan**

---

## 1. Pengertian

**Particle Swarm Optimization (PSO)** adalah algoritma optimasi berbasis swarm intelligence yang terinspirasi dari perilaku kolektif kawanan burung atau ikan. Setiap partikel mewakili satu solusi kandidat (satu set hyperparameter MLP) yang bergerak dalam ruang solusi.

> **Tujuan:** Menemukan kombinasi hyperparameter MLP $(h_1, h_2, lr, \alpha)$ yang memaksimalkan F1-macro classifier pada fitur yang telah diseleksi oleh GA.

---

## 2. Representasi Partikel

Setiap **partikel** adalah vektor posisi 4-dimensi yang merepresentasikan satu konfigurasi hyperparameter MLP:

$$\mathbf{x}_i = [h_1,\ h_2,\ lr,\ \alpha]$$

| Dimensi | Variabel | Rentang | Tipe |
|---------|----------|---------|------|
| 0 | $h_1$ — neuron hidden layer 1 | $[32, 160]$ | Integer |
| 1 | $h_2$ — neuron hidden layer 2 | $[16, 128]$ | Integer |
| 2 | $lr$ — learning rate | $[0.0005, 0.01]$ | Float |
| 3 | $\alpha$ — L2 regularization | $[0.0001, 0.01]$ | Float |

---

## 3. Fungsi Fitness PSO

$$f(\mathbf{x}_i) = \overline{F1\_macro}_{CV=3}(h_1, h_2, lr, \alpha)$$

Setiap partikel dievaluasi dengan membuat `MLPClassifier(hidden_layer_sizes=(h_1, h_2), learning_rate_init=lr, alpha=α)` dan menjalankan 3-fold cross-validation pada fitur GA-selected.

---

## 4. Persamaan Update PSO

### 4.1 Update Kecepatan

$$\mathbf{v}_i(t+1) = w \cdot \mathbf{v}_i(t) + c_1 \cdot r_1 \cdot (\mathbf{p}_{best,i} - \mathbf{x}_i(t)) + c_2 \cdot r_2 \cdot (\mathbf{g}_{best} - \mathbf{x}_i(t))$$

| Simbol | Nama | Nilai | Deskripsi |
|--------|------|-------|-----------|
| $w$ | Inertia weight | **0.7** | Bobot kecepatan sebelumnya (momentum) |
| $c_1$ | Cognitive coefficient | **1.5** | Tarikan ke personal best partikel |
| $c_2$ | Social coefficient | **1.5** | Tarikan ke global best swarm |
| $r_1, r_2$ | Random numbers | $\in [0,1]$ | Faktor stokastik (berbeda tiap iterasi) |
| $\mathbf{p}_{best,i}$ | Personal best | — | Posisi terbaik yang pernah dicapai partikel $i$ |
| $\mathbf{g}_{best}$ | Global best | — | Posisi terbaik dari seluruh swarm |

**Komponen Fisik:**

$$\underbrace{w \cdot \mathbf{v}_i(t)}_{\text{Inersia}} + \underbrace{c_1 r_1 (\mathbf{p}_{best,i} - \mathbf{x}_i)}_{\text{Kognitif (memori diri)}} + \underbrace{c_2 r_2 (\mathbf{g}_{best} - \mathbf{x}_i)}_{\text{Sosial (kolaborasi swarm)}}$$

### 4.2 Update Posisi

$$\mathbf{x}_i(t+1) = \text{clip}\left(\mathbf{x}_i(t) + \mathbf{v}_i(t+1),\ \mathbf{lb},\ \mathbf{ub}\right)$$

Di mana `clip` memastikan posisi tetap dalam batas:
$$\mathbf{lb} = [32, 16, 0.0005, 0.0001], \quad \mathbf{ub} = [160, 128, 0.01, 0.01]$$

### 4.3 Update Personal Best

$$\mathbf{p}_{best,i}(t+1) = \begin{cases} \mathbf{x}_i(t+1) & \text{jika } f(\mathbf{x}_i(t+1)) > f(\mathbf{p}_{best,i}(t)) \\ \mathbf{p}_{best,i}(t) & \text{sebaliknya} \end{cases}$$

### 4.4 Update Global Best

$$\mathbf{g}_{best}(t+1) = \arg\max_{\mathbf{x} \in \{p_{best,1}, \ldots, p_{best,N}\}} f(\mathbf{x})$$

---

## 5. Visualisasi Proses

```
Iterasi 0:  gbest_fitness = 0.74  ●→ ● ● ● ● ● ● ●  (8 partikel, posisi acak)
Iterasi 1:  gbest_fitness = 0.76  Partikel mulai mendekat ke region fitness tinggi
Iterasi 2:  gbest_fitness = 0.77  ●●●●→gbest←●●●●  konvergensi mulai
...
Iterasi 8:  gbest_fitness = 0.779 Swarm berkonvergensi ke hyperparameter optimal
```

---

## 6. Dekode Posisi ke Hyperparameter MLP

```python
def _decode(pos):
    return {
        "h1":    int(clip(round(pos[0]), 32, 160)),   # neuron layer 1
        "h2":    int(clip(round(pos[1]), 16, 128)),   # neuron layer 2
        "lr":    float(clip(pos[2], 0.0005, 0.01)),   # learning rate
        "alpha": float(clip(pos[3], 0.0001, 0.01)),   # L2 regularization
    }
```

**Contoh output:**
```python
best_params = {"h1": 96, "h2": 64, "lr": 0.0023, "alpha": 0.0045}
# MLP yang dipakai: MLPClassifier(hidden_layer_sizes=(96, 64), 
#                                  learning_rate_init=0.0023, alpha=0.0045)
```

---

## 7. Parameter yang Digunakan

| Parameter | Nilai | Keterangan |
|-----------|-------|-----------|
| `n_particles` | 8 | Ukuran swarm |
| `iterations` | 8 | Jumlah iterasi |
| `w` | 0.7 | Inertia weight |
| `c1` | 1.5 | Cognitive coefficient |
| `c2` | 1.5 | Social coefficient |
| `random_state` | 42 | Seed |

---

## 8. Pseudocode Lengkap

```
INPUT: X_train_ga (GA-selected features), y_train, n_particles, iterations

Inisialisasi posisi x_i secara uniform dalam [lb, ub] untuk i=1..N
Inisialisasi kecepatan v_i = 0
pbest_i = x_i untuk semua i
Evaluasi f(x_i) untuk semua i
gbest = argmax_i f(pbest_i)

FOR t = 1 TO iterations:
    FOR setiap partikel i:
        r1, r2 ~ Uniform(0,1)
        v_i = w*v_i + c1*r1*(pbest_i - x_i) + c2*r2*(gbest - x_i)
        x_i = clip(x_i + v_i, lb, ub)
        
        score = F1_macro_CV3(decode(x_i))
        IF score > f(pbest_i):
            pbest_i = x_i
            IF score > f(gbest):
                gbest = x_i
    
    record convergence_curve[t] = f(gbest)

RETURN decode(gbest)  // hyperparameter optimal
```

---

## 9. Cara Menggunakan Modul

```python
from modules.pso_optimizer import PSOHyperparameterOptimizer

# Inisialisasi
pso = PSOHyperparameterOptimizer(
    n_particles=8, iterations=8,
    w=0.7, c1=1.5, c2=1.5, random_state=42
)

# Training pada fitur GA-selected
pso.fit(X_train_ga, y_train)

# Ambil hyperparameter terbaik
params = pso.best_params
print("Hyperparameter optimal:", params)
# Output: {'h1': 96, 'h2': 64, 'lr': 0.0023, 'alpha': 0.0045}
print("Fitness terbaik:", pso.global_best_score_)
print("Kurva konvergensi:", pso.convergence_curve_)
```

---

## 10. Cara Membuka Hasil dari pipeline_artifacts.pkl

```python
import sys, joblib
from kelompok_x4_hybrid_pipeline import CNN1DClassifier, CNN1DNet
sys.modules["__main__"].CNN1DClassifier = CNN1DClassifier
sys.modules["__main__"].CNN1DNet = CNN1DNet

arts = joblib.load("pipeline_artifacts.pkl")

# Data PSO
print("Hyperparameter optimal:", arts["pso_params"])     # dict {h1, h2, lr, alpha}
print("Fitness terbaik:", arts["pso_best_fitness"])      # float
print("Kurva konvergensi:", arts["pso_curve"])           # list float per iterasi
```

---

## 11. Referensi

- Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. *Proceedings of ICNN'95*, 4, 1942–1948.
- Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer. *IEEE Congress on Evolutionary Computation*, 69–73.
- Clerc, M., & Kennedy, J. (2002). The particle swarm—explosion, stability, and convergence in a multidimensional complex space. *IEEE Transactions on Evolutionary Computation*, 6(1), 58–73.

---
*Kelompok X4 | TE2514031 | Institut Teknologi Kalimantan | 2026/2027*
