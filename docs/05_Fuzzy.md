# 📄 Dokumentasi: Fuzzy Inference System — Sistem Keputusan Inspeksi
**Kelompok X4 | TE2514031 | Institut Teknologi Kalimantan**

---

## 1. Pengertian

**Fuzzy Inference System (FIS)** adalah sistem pengambilan keputusan yang menggunakan logika fuzzy — yakni logika yang memungkinkan nilai kebenaran bersifat kontinu antara 0 dan 1, bukan hanya benar/salah (crisp). FIS cocok untuk masalah yang melibatkan ketidakpastian dan pengetahuan linguistik seperti "kepercayaan rendah" atau "area cacat besar".

Dalam proyek ini, FIS berfungsi sebagai **lapisan keputusan** setelah klasifikasi:

> **Input:** Kelas cacat terprediksi + confidence model + ukuran area cacat  
> **Output:** Prioritas inspeksi (Accept / Monitor / Rework / Reject) + Risk Score

---

## 2. Input Sistem Fuzzy

| Input | Simbol | Rentang | Deskripsi |
|-------|--------|---------|-----------|
| Confidence model | $p$ | $[0, 1]$ | Probabilitas softmax tertinggi dari classifier |
| Normalized area | $A$ | $[0, 1]$ | $(LogOfAreas - \min) / (\max - \min)$ |
| Kelas cacat | — | string | Nama kelas terprediksi (7 kelas) |

---

## 3. Fungsi Keanggotaan (Membership Functions)

### 3.1 Fungsi Trapesium — trapmf

$$\mu_{trap}(x;\ a, b, c, d) = \begin{cases}
0 & x \leq a \text{ atau } x \geq d \\
\dfrac{x - a}{b - a} & a < x < b \quad \text{(ramp naik)} \\
1 & b \leq x \leq c \quad \text{(plateau)} \\
\dfrac{d - x}{d - c} & c < x < d \quad \text{(ramp turun)}
\end{cases}$$

```
  Derajat
  1.0   ┤     ┌──────┐
        │    /        \
  0.0 ──┤───/──────────\──── x
         a   b    c    d
```

### 3.2 Fungsi Segitiga — trimf

$$\mu_{tri}(x;\ a, b, c) = \begin{cases}
0 & x \leq a \text{ atau } x \geq c \\
\dfrac{x - a}{b - a} & a < x \leq b \quad \text{(sisi kiri)} \\
\dfrac{c - x}{c - b} & b < x < c \quad \text{(sisi kanan)}
\end{cases}$$

```
  Derajat
  1.0   ┤       /\
        │      /  \
  0.0 ──┤─────/────\──── x
         a    b    c
```

---

## 4. Fuzzifikasi Input

### 4.1 Confidence ($p$) → 3 Set Fuzzy

| Label | Fungsi | Parameter (a, b, c, d) |
|-------|--------|------------------------|
| `prob_low` (rendah) | trapmf | (0.0, 0.0, **0.35**, **0.55**) |
| `prob_med` (sedang) | trimf | (**0.40**, **0.65**, **0.85**) |
| `prob_high` (tinggi) | trapmf | (**0.70**, **0.85**, 1.0, 1.0) |

```
Degree
1.0  ████                    ████████
     ████\         /\       /████████
0.0 ──────\───────/──\─────/──────── p
     [low]  0.35  0.55  0.70 [high]
                   [med]
```

### 4.2 Normalized Area ($A$) → 3 Set Fuzzy

| Label | Fungsi | Parameter |
|-------|--------|-----------|
| `area_small` (kecil) | trapmf | (0.0, 0.0, **0.25**, **0.45**) |
| `area_mod` (sedang) | trimf | (**0.30**, **0.55**, **0.75**) |
| `area_large` (besar) | trapmf | (**0.65**, **0.80**, 1.0, 1.0) |

---

## 5. Bobot Risiko per Kelas Cacat

Setiap kelas cacat memiliki **bobot risiko struktural** $w_c \in [0, 1]$ yang merefleksikan tingkat bahaya pada kualitas produk:

| Kelas | $w_c$ | Tingkat Bahaya |
|-------|-------|----------------|
| Stains | 0.35 | Rendah |
| Dirtiness | 0.40 | Rendah |
| Other_Faults | 0.55 | Sedang |
| Bumps | 0.70 | Tinggi |
| Pastry | 0.75 | Tinggi |
| Z_Scratch | 0.88 | Sangat Tinggi |
| K_Scatch | **0.95** | **Kritis** |

---

## 6. Basis Aturan Fuzzy (Rule Base)

Sistem menggunakan operator **min** untuk rule firing (Mamdani style):

### Aturan 1 — Accept (Terima)
$$R_{accept}: \text{IF } p \text{ rendah AND } A \text{ kecil THEN risiko rendah}$$
$$\alpha_{accept} = \min(\mu_{low}(p),\ \mu_{small}(A)) \times (1.0 - 0.5 \cdot w_c)$$

### Aturan 2 — Monitor (Pantau)
$$R_{monitor}: \text{IF } (p \text{ sedang AND } A \text{ kecil) OR } (p \text{ rendah AND } A \text{ sedang)}$$
$$\alpha_{monitor} = \max\left[\min(\mu_{med}(p), \mu_{small}(A)),\ \min(\mu_{low}(p), \mu_{mod}(A))\right]$$

### Aturan 3 — Rework (Perbaiki)
$$R_{rework}: \text{IF } (p \text{ tinggi AND } A \text{ sedang) OR } (p \text{ sedang AND } A \text{ besar)}$$
$$\alpha_{rework} = \max\left[\min(\mu_{high}(p), \mu_{mod}(A)),\ \min(\mu_{med}(p), \mu_{large}(A))\right] \times w_c$$

### Aturan 4 — Reject (Tolak)
$$R_{reject}: \text{IF } p \text{ tinggi AND } A \text{ besar THEN risiko sangat tinggi}$$
$$\alpha_{reject} = \min(\mu_{high}(p),\ \mu_{large}(A)) \times (0.6 + 0.4 \cdot w_c)$$

---

## 7. Defuzzifikasi — Risk Score

Risk score dihitung sebagai kombinasi linier terbobot:

$$\text{risk} = 0.40 \cdot p + 0.30 \cdot A + 0.30 \cdot w_c$$

**Penyesuaian berbasis kekuatan aturan:**

$$\text{risk} = \begin{cases}
\max(\text{risk},\ 0.82) & \text{jika } \alpha_{reject} > 0.5 \\
\min(\text{risk},\ 0.25) & \text{jika } \alpha_{accept} > 0.5 \text{ dan } w_c \leq 0.45 \\
\text{risk} & \text{sebaliknya}
\end{cases}$$

---

## 8. Keputusan Akhir (Crisp Output)

Risk score dikonversi ke label keputusan melalui threshold:

$$\text{decision} = \begin{cases}
\text{"Accept"} & \text{risk} < 0.30 \\
\text{"Monitor"} & 0.30 \leq \text{risk} < 0.55 \\
\text{"Rework"} & 0.55 \leq \text{risk} < 0.78 \\
\text{"Reject"} & \text{risk} \geq 0.78
\end{cases}$$

```
  Decision:  Accept │   Monitor    │    Rework    │  Reject
             ───────┼──────────────┼──────────────┼────────►  risk
                  0.30          0.55           0.78    1.0
```

---

## 9. Contoh Kalkulasi Manual

**Input:**  
- Kelas: `K_Scatch` → $w_c = 0.95$  
- Confidence: $p = 0.85$  
- Norm Area: $A = 0.72$

**Fuzzifikasi:**

```
prob_high = trapmf(0.85, 0.70, 0.85, 1.0, 1.0) = 1.0
area_large = trapmf(0.72, 0.65, 0.80, 1.0, 1.0) = (0.72 - 0.65)/(0.80 - 0.65) = 0.467
```

**Rule Reject:**

$$\alpha_{reject} = \min(1.0, 0.467) \times (0.6 + 0.4 \times 0.95) = 0.467 \times 0.98 = 0.457$$

**Risk Score:**

$$\text{risk} = 0.40 \times 0.85 + 0.30 \times 0.72 + 0.30 \times 0.95$$
$$= 0.34 + 0.216 + 0.285 = 0.841$$

**Penyesuaian:** $\alpha_{reject} = 0.457 \leq 0.5$ → tidak ada penyesuaian

**Keputusan:** $\text{risk} = 0.841 \geq 0.78$ → **"Reject"** ✅

---

## 10. Cara Menggunakan Modul

```python
from modules.fuzzy_system import SteelInspectionFuzzySystem

fuzzy = SteelInspectionFuzzySystem()

decision, risk = fuzzy.evaluate_decision(
    predicted_class_name = "K_Scatch",
    confidence_prob      = 0.85,
    normalized_area      = 0.72,
)
print(f"Keputusan: {decision}")     # Reject
print(f"Risk Score: {risk:.4f}")   # 0.8410
```

---

## 11. Cara Membuka Hasil dari pipeline_artifacts.pkl

```python
import sys, joblib
from kelompok_x4_hybrid_pipeline import CNN1DClassifier, CNN1DNet
sys.modules["__main__"].CNN1DClassifier = CNN1DClassifier
sys.modules["__main__"].CNN1DNet = CNN1DNet

arts = joblib.load("pipeline_artifacts.pkl")

# Hasil fuzzy pada 200 sampel test
fuzzy_df = arts["fuzzy_results_df"]
print(fuzzy_df.head())
print(fuzzy_df["Inspection_Priority"].value_counts())

# Kolom yang tersedia:
# Sample_ID, True_Class, Predicted_Class,
# Confidence, LogOfAreas_Raw, LogOfAreas_Norm,
# Fuzzy_Risk_Score, Inspection_Priority
```

---

## 12. Referensi

- Zadeh, L.A. (1965). Fuzzy sets. *Information and Control*, 8(3), 338–353.
- Mamdani, E.H., & Assilian, S. (1975). An experiment in linguistic synthesis with a fuzzy logic controller. *International Journal of Man-Machine Studies*, 7(1), 1–13.
- Ross, T.J. (2010). *Fuzzy Logic with Engineering Applications* (3rd ed.). Wiley.

---
*Kelompok X4 | TE2514031 | Institut Teknologi Kalimantan | 2026/2027*
