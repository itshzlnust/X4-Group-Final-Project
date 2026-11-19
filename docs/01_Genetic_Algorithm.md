# 📄 Dokumentasi: Genetic Algorithm (GA) — Seleksi Fitur
**Kelompok X4 | TE2514031 | Institut Teknologi Kalimantan**

---

## 1. Pengertian

**Genetic Algorithm (GA)** adalah metode optimasi berbasis evolusi yang terinspirasi dari proses seleksi alam Darwin. Dalam proyek ini, GA digunakan untuk **seleksi fitur** — memilih subset optimal dari 27 fitur dataset baja agar model lebih efisien dan akurat.

> **Tujuan:** Menemukan subset fitur $S \subseteq \{1, \ldots, 27\}$ yang memaksimalkan F1-macro classifier pada data latih.

---

## 2. Representasi Kromosom

Setiap **individu** (solusi kandidat) direpresentasikan sebagai **vektor biner** sepanjang 27 (jumlah fitur):

$$c = [c_1, c_2, \ldots, c_{27}], \quad c_i \in \{0, 1\}$$

| Bit | Arti |
|-----|------|
| $c_i = 1$ | Fitur ke-$i$ **dipilih** (dipakai oleh model) |
| $c_i = 0$ | Fitur ke-$i$ **dibuang** |

**Contoh kromosom:**
```
[1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
 ↑                                                                               ↑
 X_Min dipilih                                                       SigmoidOfAreas dipilih
```

**Constraint:** Minimum `min_features = 3` bit harus bernilai 1 (diperbaiki saat mutasi).

---

## 3. Fungsi Fitness

$$\text{fitness}(c) = \overline{F1\_macro}_{CV=3}(c) - \lambda \cdot \frac{|c|}{n}$$

| Simbol | Keterangan |
|--------|-----------|
| $\overline{F1\_macro}_{CV=3}(c)$ | Rata-rata F1-macro dari 3-fold cross-validation menggunakan fitur yang dipilih oleh $c$ |
| $\lambda$ | Penalti koefisien fitur = **0.001** |
| $\lvert c \rvert$ | Jumlah fitur aktif ($\sum c_i$) |
| $n$ | Total fitur = **27** |

**Interpretasi:** Semakin sedikit fitur yang dipilih dan semakin tinggi F1-macro, semakin baik fitness-nya.

---

## 4. Proses GA — Alur Generasi

### 4.1 Inisialisasi Populasi

$$P_0 = \{c_1^{(0)}, c_2^{(0)}, \ldots, c_N^{(0)}\}$$

- $N$ = `pop_size` = 12 individu
- Setiap bit diinisialisasi secara acak (Bernoulli 0/1)
- **Repair:** Jika $\sum c_i < 3$, bit-bit 0 dipilih secara acak dan di-flip ke 1

### 4.2 Evaluasi Fitness

Setiap individu $c_j$ dievaluasi menggunakan fungsi fitness di atas dengan MLPClassifier (64 neuron, 3-fold CV).

### 4.3 Seleksi — Binary Tournament

Untuk memilih satu orang tua:

$$\text{parent} = \arg\max_{j \in \{idx_1, idx_2\}} \text{fitness}(c_j)$$

Di mana $idx_1, idx_2$ dipilih secara acak tanpa pengembalian dari populasi.

> **Bug Fix #1:** Implementasi asli menggunakan dua `rng.choice` independen sehingga perbandingan fitness tidak sesuai dengan individu yang diambil. Diperbaiki dengan satu penarikan `size=2, replace=False`.

### 4.4 Crossover — Single-Point

Dua orang tua $p_1, p_2$ menghasilkan satu anak:

$$c_{cut} = p_1[0:k] \;\|\; p_2[k:27]$$

Di mana titik potong $k$ dipilih secara uniform dari $\{1, 2, \ldots, 26\}$:

```
p1 = [1,0,1,1 | 0,1,0,1,...] 
p2 = [0,1,0,0 | 1,0,1,0,...]
                ↑ cut = 4
child = [1,0,1,1 | 1,0,1,0,...]
```

### 4.5 Mutasi — Bit-Flip

Setiap gen di-flip dengan probabilitas $p_m$ = `mutation_rate` = 0.1:

$$c'_i = 1 - c_i \quad \text{dengan probabilitas } p_m$$

**Repair setelah mutasi:**
$$\text{Jika } \sum c'_i < 3: \text{ pilih bit-0 secara acak dan flip ke 1}$$

### 4.6 Elitisme

Individu terbaik dari generasi $t$ selalu diwariskan ke generasi $t+1$:

$$c_{\text{elite}}^{(t+1)} = \arg\max_{c \in P^{(t)}} \text{fitness}(c)$$

---

## 5. Parameter yang Digunakan

| Parameter | Nilai | Keterangan |
|-----------|-------|-----------|
| `n_features` | 27 | Panjang kromosom |
| `pop_size` | 12 | Ukuran populasi |
| `generations` | 10 | Jumlah iterasi |
| `mutation_rate` | 0.1 | Probabilitas mutasi per gen |
| `min_features` | 3 | Minimum fitur aktif |
| `random_state` | 42 | Seed |
| GA Runs | 3 | Dijalankan 3 kali, ambil kromosom terbaik |

---

## 6. Pseudocode Lengkap

```
INPUT: X_train (scaled), y_train, pop_size, generations, mutation_rate

Inisialisasi populasi P dengan N kromosom biner acak
Pastikan setiap kromosom memiliki >= 3 bit aktif

FOR gen = 1 TO generations:
    FOR setiap individu c dalam P:
        fitness[c] = F1_macro_CV3(c, X_train, y_train) - 0.001 * (|c|/27)
    
    Simpan best_chromosome = argmax(fitness)
    Simpan best_fitness = max(fitness)
    
    new_pop = [best_chromosome]  // elitisme
    
    WHILE |new_pop| < pop_size:
        p1 = tournament_select(P, fitness)
        p2 = tournament_select(P, fitness)
        child = single_point_crossover(p1, p2)
        child = bit_flip_mutate(child, rate=0.1)
        repair(child, min_features=3)
        new_pop.append(child)
    
    P = new_pop

RETURN best_chromosome, best_fitness
```

---

## 7. Cara Menggunakan Modul

```python
from modules.ga_feature_selector import GeneticFeatureSelector

# Inisialisasi
ga = GeneticFeatureSelector(
    n_features=27, pop_size=12, generations=10,
    mutation_rate=0.1, min_features=3, random_state=42
)

# Training (jalankan GA)
ga.fit(X_train_scaled, y_train)

# Ambil fitur terpilih
selected = ga.selected_feature_names(ALL_FEATURES)
print("Fitur terpilih:", selected)
print("Fitness terbaik:", ga.best_fitness_)
print("Kurva konvergensi:", ga.convergence_curve_)

# Subset fitur untuk model selanjutnya
X_train_ga = X_train_scaled[:, ga.best_chromosome_ == 1]
```

---

## 8. Cara Membuka Hasil dari pipeline_artifacts.pkl

```python
import sys, joblib
from kelompok_x4_hybrid_pipeline import CNN1DClassifier, CNN1DNet
sys.modules["__main__"].CNN1DClassifier = CNN1DClassifier
sys.modules["__main__"].CNN1DNet = CNN1DNet

arts = joblib.load("pipeline_artifacts.pkl")

# Data GA
print("Kromosom terbaik:", arts["ga_chromosome"])         # array binary (27,)
print("Fitur terpilih:", arts["ga_selected_features"])   # list nama fitur
print("Frekuensi fitur:", arts["ga_feature_freq"])       # dict {nama: frekuensi}
print("Kurva konvergensi:", arts["ga_curve"])            # list float per generasi
print("Semua GA runs:", arts["ga_all_results"])          # list dict hasil tiap run
```

---

## 9. Referensi

- Holland, J.H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.
- Goldberg, D.E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.
- Dash, M., & Liu, H. (1997). Feature selection for classification. *Intelligent Data Analysis*, 1(1–4), 131–156.

---
*Kelompok X4 | TE2514031 | Institut Teknologi Kalimantan | 2026/2027*
