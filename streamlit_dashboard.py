"""
streamlit_dashboard.py
======================
Multi-page Streamlit dashboard for Kelompok X4 – Steel Fault Detection.

Requirements:
    pip install streamlit scikit-learn pandas numpy matplotlib seaborn joblib

Run:
    streamlit run streamlit_dashboard.py

pipeline_artifacts.pkl must exist (run kelompok_x4_hybrid_pipeline.py or Notebook cell 13 first).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

import sys

try:
    import kelompok_x4_hybrid_pipeline
    from kelompok_x4_hybrid_pipeline import CNN1DClassifier, CNN1DNet
    sys.modules["__main__"].CNN1DClassifier = CNN1DClassifier
    sys.modules["__main__"].CNN1DNet = CNN1DNet
except (ImportError, AttributeError):
    CNN1DClassifier = None
    CNN1DNet = None

warnings.filterwarnings("ignore")

# Page config – MUST be first Streamlit call
st.set_page_config(
    page_title="Steel Fault AI – Kelompok X4",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS – steelblue / industrial palette
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #1A365D 0%, #2B6CB0 100%);
        color: #FFFFFF;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
    }
    .info-box {
        background-color: var(--secondary-background-color, rgba(44, 82, 130, 0.15));
        border: 1px solid rgba(70, 130, 180, 0.35);
        border-left: 5px solid #3B82F6;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin: 0.8rem 0;
        color: var(--text-color, inherit);
        line-height: 1.75;
    }
    .info-box b, .info-box strong {
        color: #60A5FA;
    }
    .priority-accept  {
        background: rgba(46, 204, 113, 0.18);
        border-left: 4px solid #2ECC71;
        color: #2ECC71;
        padding: 6px 14px;
        border-radius: 6px;
        display: inline-block;
        font-weight: 600;
    }
    .priority-monitor {
        background: rgba(243, 156, 18, 0.18);
        border-left: 4px solid #F1C40F;
        color: #F1C40F;
        padding: 6px 14px;
        border-radius: 6px;
        display: inline-block;
        font-weight: 600;
    }
    .priority-rework  {
        background: rgba(230, 126, 34, 0.18);
        border-left: 4px solid #E67E22;
        color: #E67E22;
        padding: 6px 14px;
        border-radius: 6px;
        display: inline-block;
        font-weight: 600;
    }
    .priority-reject  {
        background: rgba(231, 76, 60, 0.18);
        border-left: 4px solid #E74C3C;
        color: #E74C3C;
        padding: 6px 14px;
        border-radius: 6px;
        display: inline-block;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_FEATURES: List[str] = [
    "X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum",
    "Pixels_Areas", "X_Perimeter", "Y_Perimeter",
    "Sum_of_Luminosity", "Minimum_of_Luminosity", "Maximum_of_Luminosity",
    "Length_of_Conveyer", "TypeOfSteel_A300", "TypeOfSteel_A400",
    "Steel_Plate_Thickness", "Edges_Index", "Empty_Index", "Square_Index",
    "Outside_X_Index", "Edges_X_Index", "Edges_Y_Index", "Outside_Global_Index",
    "LogOfAreas", "Log_X_Index", "Log_Y_Index",
    "Orientation_Index", "Luminosity_Index", "SigmoidOfAreas",
]
TARGET_CLASSES: List[str] = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
ARTIFACT_FILE  = "pipeline_artifacts.pkl"
DATA_DIR       = Path("data")

# Feature input defaults for real-time prediction
FEAT_CFG: Dict[str, Dict] = {
    "X_Minimum":             {"min": 0.0,  "max": 1700.0,   "default": 50.0,     "step": 1.0},
    "X_Maximum":             {"min": 0.0,  "max": 1700.0,   "default": 100.0,    "step": 1.0},
    "Y_Minimum":             {"min": 0.0,  "max": 7000000.0,"default": 270000.0, "step": 100.0},
    "Y_Maximum":             {"min": 0.0,  "max": 7000000.0,"default": 271000.0, "step": 100.0},
    "Pixels_Areas":          {"min": 0.0,  "max": 150000.0, "default": 500.0,    "step": 10.0},
    "X_Perimeter":           {"min": 0.0,  "max": 1000.0,   "default": 50.0,     "step": 1.0},
    "Y_Perimeter":           {"min": 0.0,  "max": 1000.0,   "default": 50.0,     "step": 1.0},
    "Sum_of_Luminosity":     {"min": 0.0,  "max": 2.5e7,    "default": 100000.0, "step": 1000.0},
    "Minimum_of_Luminosity": {"min": 0.0,  "max": 255.0,    "default": 76.0,     "step": 1.0},
    "Maximum_of_Luminosity": {"min": 0.0,  "max": 255.0,    "default": 150.0,    "step": 1.0},
    "Length_of_Conveyer":    {"min": 0.0,  "max": 1700.0,   "default": 1200.0,   "step": 10.0},
    "TypeOfSteel_A300":      {"min": 0.0,  "max": 1.0,      "default": 1.0,      "step": 1.0},
    "TypeOfSteel_A400":      {"min": 0.0,  "max": 1.0,      "default": 0.0,      "step": 1.0},
    "Steel_Plate_Thickness": {"min": 0.0,  "max": 300.0,    "default": 80.0,     "step": 5.0},
    "Edges_Index":           {"min": 0.0,  "max": 1.0,      "default": 0.5,      "step": 0.01},
    "Empty_Index":           {"min": 0.0,  "max": 1.0,      "default": 0.35,     "step": 0.01},
    "Square_Index":          {"min": 0.0,  "max": 1.0,      "default": 0.2,      "step": 0.01},
    "Outside_X_Index":       {"min": 0.0,  "max": 1.0,      "default": 0.005,    "step": 0.001},
    "Edges_X_Index":         {"min": 0.0,  "max": 1.0,      "default": 0.47,     "step": 0.01},
    "Edges_Y_Index":         {"min": 0.0,  "max": 1.0,      "default": 0.95,     "step": 0.01},
    "Outside_Global_Index":  {"min": 0.0,  "max": 1.0,      "default": 1.0,      "step": 0.01},
    "LogOfAreas":            {"min": 0.0,  "max": 10.0,     "default": 2.43,     "step": 0.01},
    "Log_X_Index":           {"min": 0.0,  "max": 5.0,      "default": 0.9,      "step": 0.01},
    "Log_Y_Index":           {"min": 0.0,  "max": 5.0,      "default": 1.6,      "step": 0.01},
    "Orientation_Index":     {"min": -1.0, "max": 1.0,      "default": 0.82,     "step": 0.01},
    "Luminosity_Index":      {"min": -1.0, "max": 0.0,      "default": -0.29,    "step": 0.01},
    "SigmoidOfAreas":        {"min": 0.0,  "max": 1.0,      "default": 0.58,     "step": 0.01},
}

# ---------------------------------------------------------------------------
# Data + artifact loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Memuat model dan artefak …")
def load_artifacts() -> Optional[Dict[str, Any]]:
    if not os.path.exists(ARTIFACT_FILE):
        return None
    return joblib.load(ARTIFACT_FILE)


@st.cache_data(show_spinner="Memuat dataset dari file lokal …")
def load_raw_data() -> pd.DataFrame:
    all_col_names = ALL_FEATURES + TARGET_CLASSES
    df = pd.read_csv(DATA_DIR / "Faults.NNA", sep=r"\s+", header=None, names=all_col_names)
    return df


# ---------------------------------------------------------------------------
# Fuzzy helper (inline – no dependency on pipeline module)
# ---------------------------------------------------------------------------
SEVERITY_MAP: Dict[str, float] = {
    "Stains": 0.35, "Dirtiness": 0.40, "Other_Faults": 0.55,
    "Bumps": 0.70,  "Pastry": 0.75,    "Z_Scratch": 0.88, "K_Scatch": 0.95,
}

def _trapmf(x, a, b, c, d):
    eps = 1e-9
    if x <= a or x >= d: return 0.0
    if b <= x <= c:      return 1.0
    if a < x < b:        return (x - a) / max(b - a, eps)
    return (d - x) / max(d - c, eps)

def _trimf(x, a, b, c):
    eps = 1e-9
    if x <= a or x >= c: return 0.0
    if x <= b: return (x - a) / max(b - a, eps)
    return (c - x) / max(c - b, eps)

def fuzzy_decision(pred_cls: str, conf: float, norm_area: float):
    crit      = SEVERITY_MAP.get(pred_cls, 0.55)
    prob_low  = _trapmf(conf, 0.0, 0.0, 0.35, 0.55)
    prob_high = _trapmf(conf, 0.70, 0.85, 1.0, 1.0)
    area_sm   = _trapmf(norm_area, 0.0, 0.0, 0.25, 0.45)
    area_lg   = _trapmf(norm_area, 0.65, 0.80, 1.0, 1.0)
    r_accept  = min(prob_low, area_sm) * (1.0 - 0.5 * crit)
    r_reject  = min(prob_high, area_lg) * (0.6 + 0.4 * crit)
    risk = 0.40 * conf + 0.30 * norm_area + 0.30 * crit
    if r_reject > 0.5:                   risk = max(risk, 0.82)
    elif r_accept > 0.5 and crit < 0.45: risk = min(risk, 0.25)
    if   risk < 0.30: decision = "Accept"
    elif risk < 0.55: decision = "Monitor"
    elif risk < 0.78: decision = "Rework"
    else:             decision = "Reject"
    return decision, round(float(risk), 4)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
PAGES = [
    "🏠  Beranda",
    "📊  Eksplorasi Data",
    "🧬  Genetic Algorithm",
    "🌊  Particle Swarm Optimization",
    "🤖  Model Comparison",
    "🔍  Sistem Fuzzy",
    "🔮  Prediksi Real-Time",
]

with st.sidebar:
    st.markdown("## 🔩 Steel Fault AI")
    st.markdown("**Kelompok X4** | TE2514031")
    st.markdown("Institut Teknologi Kalimantan")
    st.markdown("---")
    page = st.radio("Navigasi Halaman", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.caption("Dosen: Happy Aprillia, S.ST., M.T., M.Eng., Ph.D.")
    st.caption("© Semester Gasal 2026/2027")

artifacts = load_artifacts()

def _require_artifacts() -> bool:
    if artifacts is None:
        st.warning(
            "⚠️ **Artefak pipeline belum tersedia.**\n\n"
            "Jalankan pipeline terlebih dahulu:\n"
            "```bash\npython kelompok_x4_hybrid_pipeline.py\n```\n"
            "atau jalankan semua cell di notebook **FIXED** hingga Cell 13.",
            icon="⚠️",
        )
        return False
    return True


# ===========================================================================
# PAGE 1 – BERANDA
# ===========================================================================
if page == PAGES[0]:
    st.markdown(
        """
        <div class="main-header">
            <h1 style="margin:0">🔩 Intelligent Steel Surface Fault Detection</h1>
            <p style="margin:4px 0 0 0; opacity:0.85">
                Hybrid AI: Genetic Algorithm + PSO + Fuzzy System + MLP Neural Network
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("👥 Informasi Tim")
        st.table(pd.DataFrame({
            "": ["Kelompok", "Mata Kuliah", "Dosen", "Dataset", "Jumlah Fitur", "Kelas Target"],
            "Detail": [
                "X4 (Kelas X)",
                "Kecerdasan Buatan (TE2514031)",
                "Happy Aprillia, S.ST., M.T., M.Eng., Ph.D.",
                "Steel Plates Faults – UCI ID 198 (1,941 sampel)",
                "27 (14 Integer + 13 Continuous)",
                "7 jenis cacat permukaan baja",
            ],
        }).set_index(""))

        st.subheader("🏗️ Arsitektur Hybrid Pipeline")
        st.markdown(
            """
            <div class="info-box">
                <div style="font-weight: 700; margin-bottom: 0.75rem; font-size: 1.05rem; color: #60A5FA;">Alur Pipeline:</div>
                <div style="margin-bottom: 0.25rem;">📥 <b>Input</b>: 1,941 sampel × 27 fitur | 7 kelas cacat</div>
                <div style="opacity: 0.6; padding-left: 1.5rem; margin-bottom: 0.25rem;">↓</div>
                <div style="margin-bottom: 0.25rem;">🧬 <b>Genetic Algorithm (GA)</b> – Seleksi fitur (binary chromosome, 3 independent runs)</div>
                <div style="opacity: 0.6; padding-left: 1.5rem; margin-bottom: 0.25rem;">↓ <i>Subset 14 fitur terpilih</i></div>
                <div style="margin-bottom: 0.25rem;">🌊 <b>Particle Swarm Optimization (PSO)</b> – Tuning hyperparameter MLP (h1, h2, lr, α)</div>
                <div style="opacity: 0.6; padding-left: 1.5rem; margin-bottom: 0.25rem;">↓ <i>Hyperparameter optimal</i></div>
                <div style="margin-bottom: 0.25rem;">🤖 <b>Neural Networks (MLP & 1D-CNN)</b> – 6 varian model (A/B/C/D MLP + E/F 1D-CNN PyTorch)</div>
                <div style="opacity: 0.6; padding-left: 1.5rem; margin-bottom: 0.25rem;">↓ <i>Probabilitas kelas</i></div>
                <div style="margin-bottom: 0.25rem;">🔍 <b>Fuzzy Inference System</b> – Risk Score & Inspection Priority</div>
                <div style="opacity: 0.6; padding-left: 1.5rem; margin-bottom: 0.25rem;">↓</div>
                <div>📊 <b>Streamlit Dashboard</b> – Visualisasi & Prediksi Real-Time</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.subheader("📋 Info Dataset")
        st.metric("Total Sampel", "1,941")
        st.metric("Fitur Input",  "27")
        st.metric("Kelas Target", "7")
        st.metric("Test Split",   "20%  (388 sampel)")
        st.markdown("**7 Kelas Cacat Baja:**")
        for cls in TARGET_CLASSES:
            st.markdown(f"• `{cls}`")

    if artifacts:
        st.markdown("---")
        st.success("✅ Artefak pipeline berhasil dimuat. Eksplorasi halaman lainnya di sidebar!")
        cm_df = artifacts.get("comparison_df")
        if cm_df is not None:
            best_model = cm_df["F1-Score (Macro)"].idxmax() if "F1-Score (Macro)" in cm_df.columns \
                         else cm_df.iloc[:, 3].idxmax()
            best_row   = cm_df.loc[best_model]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Best Model",    best_model.split(":")[0])
            c2.metric("F1-Macro",      f"{best_row.get('F1-Score (Macro)', 0):.4f}")
            c3.metric("Accuracy",      f"{best_row.get('Accuracy', 0):.4f}")
            sel = artifacts.get("ga_selected_features", [])
            c4.metric("GA Features",   f"{len(sel)}/27")
    else:
        st.info("ℹ️  Jalankan pipeline atau notebook untuk melatih model dan mengisi dashboard.")


# ===========================================================================
# PAGE 2 – EDA
# ===========================================================================
elif page == PAGES[1]:
    st.header("📊 Eksplorasi Data (EDA)")
    try:
        df = load_raw_data()
    except FileNotFoundError:
        st.error(f"File dataset tidak ditemukan di folder `{DATA_DIR}/`. Pastikan `Faults.NNA` tersedia.")
        st.stop()

    y_label = df[TARGET_CLASSES].idxmax(axis=1)
    X_raw   = df[ALL_FEATURES]

    tab1, tab2, tab3, tab4 = st.tabs(["Distribusi Kelas", "Heatmap Korelasi", "Box Plots", "Statistik"])

    with tab1:
        st.subheader("Distribusi 7 Kelas Cacat Baja")
        counts = y_label.value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(counts.index, counts.values,
                      color=sns.color_palette("Blues_d", len(counts)), edgecolor="white")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
                    str(int(bar.get_height())), ha="center", va="bottom", fontsize=10)
        ax.set_xlabel("Kelas Cacat", fontsize=12); ax.set_ylabel("Jumlah Sampel", fontsize=12)
        ax.set_title("Distribusi Kelas Cacat Permukaan Baja", fontsize=14, fontweight="bold")
        ax.set_facecolor("#F8FAFC"); ax.tick_params(axis="x", rotation=12)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    with tab2:
        st.subheader("Heatmap Korelasi Fitur")
        n_cols = st.slider("Jumlah fitur pertama yang ditampilkan:", 10, 27, 14)
        corr = X_raw.iloc[:, :n_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, linewidths=0.3,
                    annot=(n_cols <= 14), fmt=".2f", ax=ax, cbar_kws={"shrink": 0.8},
                    annot_kws={"size": 7})
        ax.set_title(f"Korelasi {n_cols} Fitur Pertama", fontsize=13, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    with tab3:
        st.subheader("Box Plots Fitur per Kelas")
        feat_sel = st.selectbox("Pilih fitur:", ALL_FEATURES, index=ALL_FEATURES.index("LogOfAreas"))
        df_plot  = df[[feat_sel]].copy()
        df_plot["Kelas"] = y_label.values
        fig, ax  = plt.subplots(figsize=(12, 5))
        grouped  = {cls: df_plot[df_plot["Kelas"] == cls][feat_sel].dropna().values
                    for cls in TARGET_CLASSES}
        try:
            ax.boxplot(list(grouped.values()), tick_labels=list(grouped.keys()), patch_artist=True,
                       boxprops=dict(facecolor="#B0C4DE", color="#2C4A6E"),
                       medianprops=dict(color="#E74C3C", linewidth=2))
        except TypeError:
            ax.boxplot(list(grouped.values()), labels=list(grouped.keys()), patch_artist=True,
                       boxprops=dict(facecolor="#B0C4DE", color="#2C4A6E"),
                       medianprops=dict(color="#E74C3C", linewidth=2))
        ax.set_xlabel("Kelas Cacat"); ax.set_ylabel(feat_sel)
        ax.set_title(f"Distribusi {feat_sel} per Kelas", fontsize=13, fontweight="bold")
        ax.set_facecolor("#F8FAFC"); ax.tick_params(axis="x", rotation=12)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    with tab4:
        st.subheader("Statistik Deskriptif")
        st.dataframe(X_raw.describe().T.style.format("{:.4f}"), use_container_width=True)


# ===========================================================================
# PAGE 3 – GENETIC ALGORITHM
# ===========================================================================
elif page == PAGES[2]:
    st.header("🧬 Genetic Algorithm – Seleksi Fitur")
    if not _require_artifacts(): st.stop()

    ga_all   = artifacts.get("ga_all_results", [])
    ga_curve = artifacts.get("ga_curve", [])
    sel_feat = artifacts.get("ga_selected_features", [])
    ga_freq  = artifacts.get("ga_feature_freq", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fitur Asli",      "27")
    c2.metric("Fitur Terpilih",  len(sel_feat))
    c3.metric("Reduksi Dimensi", f"{(1 - len(sel_feat)/27)*100:.1f}%")
    c4.metric("GA Runs",         len(ga_all))

    tab1, tab2, tab3 = st.tabs(["Convergence Curves", "Fitur Terpilih", "Statistik Runs"])

    with tab1:
        palette = ["#4682B4", "#E74C3C", "#2ECC71"]
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        # Per-run curves
        if ga_all:
            for d in ga_all:
                crv = d.get("curve", [])
                r   = d.get("run", 1)
                axes[0].plot(range(1, len(crv)+1), crv,
                             label=f"Run {r} (F1={d.get('best_fitness',0):.4f})",
                             color=palette[(r-1) % 3], linewidth=2, marker="o", markersize=4)
        axes[0].set_xlabel("Generasi"); axes[0].set_ylabel("Best F1-Macro Fitness")
        axes[0].set_title("GA Convergence – 3 Independent Runs", fontweight="bold")
        axes[0].legend(); axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].set_facecolor("#F8FAFC")
        # Best run fill
        if ga_curve:
            axes[1].plot(range(1, len(ga_curve)+1), ga_curve,
                         color="#2C4A6E", linewidth=2.5, marker="o", markersize=4)
            axes[1].fill_between(range(1, len(ga_curve)+1), ga_curve, alpha=0.15, color="#4682B4")
        axes[1].set_xlabel("Generasi"); axes[1].set_ylabel("Best Fitness")
        axes[1].set_title("Best Run – Convergence Detail", fontweight="bold")
        axes[1].grid(True, linestyle="--", alpha=0.5); axes[1].set_facecolor("#F8FAFC")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    with tab2:
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.subheader("Fitur Terpilih GA")
            st.dataframe(pd.DataFrame({
                "Fitur": sel_feat,
                "Frekuensi (3 runs)": [round(ga_freq.get(f, 0), 3) for f in sel_feat],
            }), use_container_width=True, height=400)
        with col_b:
            if ga_freq:
                sorted_freq = sorted(ga_freq.items(), key=lambda x: x[1], reverse=True)[:15]
                fig, ax = plt.subplots(figsize=(8, 6))
                names  = [f[0] for f in sorted_freq]
                vals   = [f[1] for f in sorted_freq]
                colors = ["#4682B4" if n in sel_feat else "#B0C4DE" for n in names]
                ax.barh(names[::-1], vals[::-1], color=colors[::-1], edgecolor="white")
                ax.set_xlabel("Frekuensi (0–1)")
                ax.set_title("Top-15 Frekuensi Seleksi Fitur", fontweight="bold")
                ax.axvline(0.5, color="#E74C3C", linestyle="--", linewidth=1.5, label="Threshold 0.5")
                ax.legend(); ax.set_facecolor("#F8FAFC")
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    with tab3:
        if ga_all:
            stats = [{"Run": d["run"],
                      "Best F1": round(d.get("best_fitness", 0), 4),
                      "Jumlah Fitur": d.get("n_features", "—")}
                     for d in ga_all]
            st.dataframe(pd.DataFrame(stats), use_container_width=True)
            fits = [d.get("best_fitness", 0) for d in ga_all]
            col_m, col_s = st.columns(2)
            col_m.metric("Mean Best Fitness", f"{np.mean(fits):.4f}")
            col_s.metric("Std  Best Fitness", f"{np.std(fits):.4f}")


# ===========================================================================
# PAGE 4 – PSO
# ===========================================================================
elif page == PAGES[3]:
    st.header("🌊 Particle Swarm Optimization – Hyperparameter Tuning")
    if not _require_artifacts(): st.stop()

    pso_params = artifacts.get("pso_params", {})
    pso_curve  = artifacts.get("pso_curve", [])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hidden Layer 1 (h1)", pso_params.get("h1", "—"))
    c2.metric("Hidden Layer 2 (h2)", pso_params.get("h2", "—"))
    c3.metric("Learning Rate",       f"{pso_params.get('lr', 0):.5f}")
    c4.metric("Alpha (L2)",          f"{pso_params.get('alpha', 0):.5f}" if "alpha" in pso_params else "—")

    tab1, tab2 = st.tabs(["Convergence Curve", "Parameter Comparison"])

    with tab1:
        if pso_curve:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(range(1, len(pso_curve)+1), pso_curve,
                    color="#2C4A6E", linewidth=2.5, marker="s", markersize=5)
            ax.fill_between(range(1, len(pso_curve)+1), pso_curve, alpha=0.12, color="#4682B4")
            ax.set_xlabel("Iterasi PSO", fontsize=12)
            ax.set_ylabel("Global Best F1-Macro (CV)", fontsize=12)
            ax.set_title("PSO Convergence – MLP Hyperparameter Search", fontsize=13, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.5); ax.set_facecolor("#F8FAFC")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        else:
            st.info("Data konvergensi PSO tidak tersedia.")

    with tab2:
        compare = pd.DataFrame({
            "Parameter":   ["h1 (Hidden Layer 1)", "h2 (Hidden Layer 2)", "Learning Rate", "Alpha (L2)"],
            "Default MLP": [100, "—", 0.001, 0.0001],
            "PSO Optimal": [
                pso_params.get("h1", "—"),
                pso_params.get("h2", "—"),
                f"{pso_params.get('lr', 0):.5f}",
                f"{pso_params.get('alpha', 0):.5f}" if "alpha" in pso_params else "—",
            ],
        })
        st.table(compare)
        st.markdown(
            """
            <div class="info-box">
            <b>Ruang Pencarian PSO (4-D):</b><br>
            &bull; <b>h1</b>: neuron hidden layer 1 → [32, 160]<br>
            &bull; <b>h2</b>: neuron hidden layer 2 → [16, 128]<br>
            &bull; <b>lr</b>: learning rate → [0.0005, 0.01]<br>
            &bull; <b>α</b>: L2 regularisasi → [0.0001, 0.01]
            </div>
            """,
            unsafe_allow_html=True,
        )


# ===========================================================================
# PAGE 5 – MODEL COMPARISON
# ===========================================================================
elif page == PAGES[4]:
    st.header("🤖 Perbandingan Model")
    if not _require_artifacts(): st.stop()

    comp_df   = artifacts.get("comparison_df")
    cm_d      = artifacts.get("confusion_matrix")
    cr_d      = artifacts.get("classification_report", "")
    cls_names = artifacts.get("class_names", TARGET_CLASSES)

    tab1, tab2, tab3 = st.tabs(["Tabel & Bar Chart", "Confusion Matrix", "Classification Report"])

    with tab1:
        st.subheader("Metrik Semua Model")
        try:
            st.dataframe(
                comp_df.style.highlight_max(axis=0, color="#C8E6C9").format("{:.4f}"),
                use_container_width=True,
            )
        except Exception:
            st.dataframe(comp_df, use_container_width=True)

        st.markdown("---")
        st.subheader("Perbandingan F1-Score (Macro)")
        f1_col = "F1-Score (Macro)" if "F1-Score (Macro)" in comp_df.columns else comp_df.columns[3]
        f1_vals = comp_df[f1_col].values
        mod_labels = comp_df.index.tolist()
        palette = ["#B0C4DE", "#7FB3D3", "#4682B4", "#2C4A6E", "#E67E22", "#27AE60"]
        bar_colors = [palette[i % len(palette)] for i in range(len(mod_labels))]
        fig, ax = plt.subplots(figsize=(11, 5))
        bars = ax.bar(range(len(mod_labels)), f1_vals, color=bar_colors, edgecolor="white")
        ax.set_xticks(range(len(mod_labels)))
        ax.set_xticklabels([m.split(":")[0] for m in mod_labels], rotation=10)
        for bar, v in zip(bars, f1_vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05); ax.set_ylabel("F1-Score (Macro)")
        ax.set_title("Perbandingan F1-Macro per Model (MLP & 1D-CNN)", fontsize=13, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5); ax.set_facecolor("#F8FAFC")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.markdown(
            """<div class="info-box"><b>Keterangan:</b><br>
            &bull; <b>Model A</b>: Baseline MLP, semua 27 fitur<br>
            &bull; <b>Model B</b>: GA feature selection + MLP default<br>
            &bull; <b>Model C</b>: PSO hyperparameter tuning + MLP (semua fitur)<br>
            &bull; <b>Model D</b>: GA + PSO + MLP (Full Hybrid MLP) ✅<br>
            &bull; <b>Model E</b>: 1D-CNN (PyTorch Conv1D), semua 27 fitur<br>
            &bull; <b>Model F</b>: GA + 1D-CNN (Hybrid 1D-CNN) 🚀</div>""",
            unsafe_allow_html=True,
        )

    with tab2:
        cm_d = artifacts.get("confusion_matrix")
        cm_f = artifacts.get("confusion_matrix_cnn")
        cm_options = ["Model D (GA + PSO + MLP)", "Model F (GA + 1D-CNN)"] if cm_f is not None else ["Model D (GA + PSO + MLP)"]
        cm_choice = st.radio("Pilih Confusion Matrix Model:", cm_options, horizontal=True)
        cm_to_show = cm_f if "1D-CNN" in cm_choice and cm_f is not None else cm_d
        title_cm = "Confusion Matrix – Model F (GA + 1D-CNN)" if "1D-CNN" in cm_choice else "Confusion Matrix – Model D (GA + PSO + MLP)"
        if cm_to_show is not None:
            labels = cls_names if len(cls_names) == cm_to_show.shape[0] else [str(i) for i in range(cm_to_show.shape[0])]
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm_to_show, annot=True, fmt="d", cmap="Blues",
                        xticklabels=labels, yticklabels=labels,
                        linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
            ax.set_xlabel("Predicted Label", fontsize=12); ax.set_ylabel("True Label", fontsize=12)
            ax.set_title(title_cm, fontsize=13, fontweight="bold")
            ax.tick_params(axis="x", rotation=15); ax.tick_params(axis="y", rotation=0)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        else:
            st.info("Confusion matrix tidak tersedia.")

    with tab3:
        cr_d = artifacts.get("classification_report", "")
        cr_f = artifacts.get("classification_report_cnn", "")
        if cr_f:
            sub1, sub2 = st.tabs(["Model D (GA + PSO + MLP)", "Model F (GA + 1D-CNN)"])
            with sub1:
                st.code(cr_d, language="text")
            with sub2:
                st.code(cr_f, language="text")
        elif cr_d:
            st.code(cr_d, language="text")


# ===========================================================================
# PAGE 6 – FUZZY SYSTEM
# ===========================================================================
elif page == PAGES[5]:
    st.header("🔍 Sistem Fuzzy – Inspection Priority")
    if not _require_artifacts(): st.stop()

    fuzzy_df = artifacts.get("fuzzy_results_df", pd.DataFrame())
    if fuzzy_df.empty:
        st.warning("Data fuzzy tidak tersedia."); st.stop()

    st.markdown(
        "<b>Legenda:</b> "
        "<span class='priority-accept'>🟢 Accept</span> &nbsp; "
        "<span class='priority-monitor'>🟡 Monitor</span> &nbsp; "
        "<span class='priority-rework'>🟠 Rework</span> &nbsp; "
        "<span class='priority-reject'>🔴 Reject</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Determine risk and priority columns
    risk_col = "Fuzzy_Risk_Score" if "Fuzzy_Risk_Score" in fuzzy_df.columns else \
               [c for c in fuzzy_df.columns if "risk" in c.lower()][0]
    pri_col  = "Inspection_Priority" if "Inspection_Priority" in fuzzy_df.columns else \
               [c for c in fuzzy_df.columns if "priority" in c.lower()][0]

    tab1, tab2, tab3, tab4 = st.tabs(["Tabel Hasil", "Distribusi Prioritas", "Risk Score", "Akurasi"])

    with tab1:
        pc_map = {
            "Accept":  "background-color: rgba(46, 204, 113, 0.25); color: #2ecc71; font-weight: 600;",
            "Monitor": "background-color: rgba(243, 156, 18, 0.25); color: #f1c40f; font-weight: 600;",
            "Rework":  "background-color: rgba(230, 126, 34, 0.25); color: #e67e22; font-weight: 600;",
            "Reject":  "background-color: rgba(231, 76, 60, 0.25); color: #e74c3c; font-weight: 600;",
        }
        def _color_pri(val):
            return pc_map.get(val, "")
        try:
            st.dataframe(fuzzy_df.style.applymap(_color_pri, subset=[pri_col]),
                         use_container_width=True, height=420)
        except Exception:
            st.dataframe(fuzzy_df, use_container_width=True, height=420)
        vc = fuzzy_df[pri_col].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accept",  int(vc.get("Accept",  0)))
        c2.metric("Monitor", int(vc.get("Monitor", 0)))
        c3.metric("Rework",  int(vc.get("Rework",  0)))
        c4.metric("Reject",  int(vc.get("Reject",  0)))

    with tab2:
        vc = fuzzy_df[pri_col].value_counts()
        pie_clr = {"Accept": "#2ECC71", "Monitor": "#F39C12", "Rework": "#E67E22", "Reject": "#E74C3C"}
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(vc.values, labels=vc.index, autopct="%1.1f%%",
               colors=[pie_clr.get(k, "#AAA") for k in vc.index],
               startangle=140, wedgeprops={"edgecolor": "white", "linewidth": 2})
        ax.set_title("Distribusi Inspection Priority", fontsize=14, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    with tab3:
        risk_vals = fuzzy_df[risk_col].values.astype(float)
        is_normalized = risk_vals.max() <= 1.0
        thresholds = (0.30, 0.55, 0.78) if is_normalized else (25, 50, 75)
        labels_thr = ("Accept/Monitor", "Monitor/Rework", "Rework/Reject")
        clrs_thr   = ("#2ECC71", "#F39C12", "#E74C3C")

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.hist(risk_vals, bins=25, color="#4682B4", edgecolor="white", alpha=0.85)
        for t, lbl, clr in zip(thresholds, labels_thr, clrs_thr):
            ax.axvline(t, color=clr, linestyle="--", linewidth=1.8, label=f"{lbl} ({t})")
        ax.set_xlabel("Fuzzy Risk Score", fontsize=12); ax.set_ylabel("Frekuensi", fontsize=12)
        ax.set_title("Distribusi Risk Score – Fuzzy System", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9); ax.set_facecolor("#F8FAFC")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean",  f"{risk_vals.mean():.4f}")
        c2.metric("Std",   f"{risk_vals.std():.4f}")
        c3.metric("Min",   f"{risk_vals.min():.4f}")
        c4.metric("Max",   f"{risk_vals.max():.4f}")

    with tab4:
        pred_col = "Predicted_Defect" if "Predicted_Defect" in fuzzy_df.columns else \
                   [c for c in fuzzy_df.columns if "predict" in c.lower()][0]
        true_col = "True_Class" if "True_Class" in fuzzy_df.columns else \
                   [c for c in fuzzy_df.columns if "true" in c.lower() or "actual" in c.lower()][0]
        correct = (fuzzy_df[true_col] == fuzzy_df[pred_col]).sum()
        total   = len(fuzzy_df)
        st.metric("Akurasi Prediksi (Fuzzy Subset)", f"{correct}/{total}  ({correct/total*100:.1f}%)")


# ===========================================================================
# PAGE 7 – REAL-TIME PREDICTION
# ===========================================================================
elif page == PAGES[6]:
    st.header("🔮 Prediksi Real-Time")
    if not _require_artifacts(): st.stop()

    scaler      = artifacts["scaler"]
    le          = artifacts["le"]
    ga_chrom    = artifacts["ga_chromosome"]
    final_model = artifacts["final_model"]
    sel_feat    = artifacts.get("ga_selected_features", [])
    cls_names   = artifacts.get("class_names", list(le.classes_))

    # LogOfAreas normalization bounds from training data
    X_test_orig = artifacts.get("X_test_orig", pd.DataFrame())
    area_min = float(X_test_orig["LogOfAreas"].min()) if not X_test_orig.empty else 1.5
    area_max = float(X_test_orig["LogOfAreas"].max()) if not X_test_orig.empty else 4.5

    # Model selection for real-time inference
    model_opts = ["Model D: GA + PSO + MLP (Full Hybrid)", "Model F: GA + 1D-CNN (PyTorch 1D-CNN)"] if "final_model_cnn" in artifacts else ["Model D: GA + PSO + MLP (Full Hybrid)"]
    selected_model_name = st.selectbox("Pilih Model untuk Inferensi:", model_opts, index=0)
    active_model = artifacts.get("final_model_cnn", final_model) if "1D-CNN" in selected_model_name else final_model

    st.markdown(
        f"""<div class="info-box">
        Model aktif: <b>{selected_model_name}</b> &nbsp;|&nbsp;
        Fitur digunakan: <b>{len(sel_feat)}/27 (GA selected)</b> &nbsp;|&nbsp;
        Kelas: <b>{', '.join(cls_names)}</b>
        </div>""",
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        st.subheader("📋 Input Nilai 27 Fitur")
        input_values: Dict[str, float] = {}
        chunks = [ALL_FEATURES[i:i+3] for i in range(0, len(ALL_FEATURES), 3)]
        for chunk in chunks:
            cols = st.columns(len(chunk))
            for col, feat in zip(cols, chunk):
                cfg = FEAT_CFG.get(feat, {"min": 0.0, "max": 100.0, "default": 0.0, "step": 0.01})
                with col:
                    input_values[feat] = st.number_input(
                        label=feat,
                        min_value=float(cfg["min"]),
                        max_value=float(cfg["max"]),
                        value=float(cfg["default"]),
                        step=float(cfg["step"]),
                        key=f"inp_{feat}",
                    )
        submitted = st.form_submit_button("🔍 Prediksi", use_container_width=True, type="primary")

    if submitted:
        with st.spinner("Memproses …"):
            x_raw = np.array([[input_values.get(f, 0.0) for f in ALL_FEATURES]], dtype=float)
            x_sc  = scaler.transform(x_raw)
            sel_idx = np.where(ga_chrom == 1)[0]
            x_ga  = x_sc[:, sel_idx]
            proba = active_model.predict_proba(x_ga)[0]
            pred_idx  = int(np.argmax(proba))
            pred_cls  = le.inverse_transform([pred_idx])[0]
            confidence = float(proba[pred_idx])
            # Fuzzy
            log_area  = float(input_values.get("LogOfAreas", 2.43))
            norm_area = float((log_area - area_min) / max(area_max - area_min, 1e-9))
            decision, risk = fuzzy_decision(pred_cls, confidence, np.clip(norm_area, 0, 1))

        st.markdown("---")
        st.subheader("🎯 Hasil Prediksi")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Jenis Cacat",  pred_cls)
        c2.metric("Confidence",   f"{confidence*100:.2f}%")
        c3.metric("Risk Score",   f"{risk:.4f}")
        c4.metric("Prioritas",    decision)

        # Confidence bar
        fig, ax = plt.subplots(figsize=(10, 3))
        bar_colors = ["#4682B4" if i == pred_idx else "#B0C4DE" for i in range(len(cls_names))]
        ax.barh(cls_names, proba, color=bar_colors, edgecolor="white")
        ax.set_xlim(0, 1); ax.set_xlabel("Probability")
        ax.set_title("Distribusi Probabilitas Kelas", fontweight="bold")
        for i, v in enumerate(proba):
            ax.text(min(v + 0.01, 0.95), i, f"{v:.3f}", va="center", fontsize=9)
        ax.set_facecolor("#F8FAFC")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        # Priority badge
        pm = {
            "Accept":  ("priority-accept",  "🟢 ACCEPT  – Produk dapat diterima"),
            "Monitor": ("priority-monitor", "🟡 MONITOR – Perlu pemantauan lanjutan"),
            "Rework":  ("priority-rework",  "🟠 REWORK  – Perlu pengerjaan ulang"),
            "Reject":  ("priority-reject",  "🔴 REJECT  – Produk ditolak / tidak lolos QC"),
        }
        css_cls, msg = pm.get(decision, ("info-box", decision))
        st.markdown(
            f'<div class="{css_cls}" style="font-size:1.1rem; font-weight:bold; '
            f'padding:0.9rem 1.4rem; margin-top:0.8rem;">{msg}</div>',
            unsafe_allow_html=True,
        )
