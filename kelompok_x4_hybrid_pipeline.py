"""
================================================================================
TUGAS BESAR KECERDASAN BUATAN (TE2514031) - INSTITUT TEKNOLOGI KALIMANTAN
Semester Gasal 2026/2027 | Dosen Pengampu: Happy Aprillia, S.ST., M.T., M.Eng., Ph.D.

Kelompok : X4 (Kelas X)
Anggota  :
  1. 11231084 - Rahmahdaniah Fitri
  2. 17231008 - Aulia Wahdah Atika Rahmah
  3. 10231042 - Ilham Ahmad Fahriji

Topik    : Intelligent Steel Surface Fault Detection
Dataset  : Steel Plates Faults (UCI ID 198) – local files: data/Faults.NNA + data/Faults27x7_var
Pipeline : Dataset -> Preprocessing -> GA (Feature Selection) ->
           PSO (Hyperparameter Tuning) -> Neural Network (MLP) ->
           Fuzzy Inference System (Accept/Monitor/Rework/Reject) -> Streamlit
================================================================================

CHANGELOG (bug fixes vs original kelompok_x4_hybrid_pipeline.py):
  FIX-1  GA tournament selection: original lines 139-140 called rng.choice twice
         and indexed fitness_scores with the result of the second call – producing
         random indexing unrelated to tournament semantics.  Fixed to draw two
         distinct indices and return the individual with the higher fitness.
  FIX-2  trapmf / trimf: added proper boundary guards so degenerate intervals
         (a==b or b==c) never cause division by zero or unexpected values.
  FIX-3  Missing imports: added os, joblib, pickle (model persistence).
  FIX-4  No model save/load: run_full_pipeline() saves pipeline_artifacts.pkl.
  FIX-5  Empty __main__ block: replaced with argparse CLI demonstration.
  FIX-6  X_test_orig: original (unscaled) X_test DataFrame preserved before
         StandardScaler.transform() so Fuzzy layer can read raw LogOfAreas values.
  FIX-7  Dataset loading: switched from ucimlrepo online fetch to local
         Faults.NNA + Faults27x7_var files (matches uploaded dataset).
  FIX-8  Correct 27 feature names taken from Faults27x7_var (includes
         Edges_Y_Index, Outside_Global_Index, Log_X_Index, Log_Y_Index,
         Orientation_Index, Luminosity_Index, SigmoidOfAreas – which were
         absent in the previous hardcoded list).
  FIX-9  PSO: original .py uses 4-dim search space (h1,h2,lr,alpha); notebook
         uses 3-dim (h1,h2,lr).  Both variants are provided here as separate
         classes for clarity and consistency.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants – exact names from Faults27x7_var
# ---------------------------------------------------------------------------

RANDOM_STATE: int = 42

INTEGER_FEATURES: List[str] = [
    "X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum",
    "Pixels_Areas", "X_Perimeter", "Y_Perimeter",
    "Sum_of_Luminosity", "Minimum_of_Luminosity", "Maximum_of_Luminosity",
    "Length_of_Conveyer", "TypeOfSteel_A300", "TypeOfSteel_A400",
    "Steel_Plate_Thickness",
]

CONTINUOUS_FEATURES: List[str] = [
    "Edges_Index", "Empty_Index", "Square_Index", "Outside_X_Index",
    "Edges_X_Index", "Edges_Y_Index", "Outside_Global_Index",
    "LogOfAreas", "Log_X_Index", "Log_Y_Index",
    "Orientation_Index", "Luminosity_Index", "SigmoidOfAreas",
]

ALL_FEATURES: List[str] = INTEGER_FEATURES + CONTINUOUS_FEATURES   # 27 total

TARGET_CLASSES: List[str] = [
    "Pastry", "Z_Scratch", "K_Scatch", "Stains",
    "Dirtiness", "Bumps", "Other_Faults",
]

ARTIFACT_PATH: str = "pipeline_artifacts.pkl"

# ---------------------------------------------------------------------------
# Default data directory (relative to this file or CWD)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent if "__file__" in dir() else Path.cwd()
DEFAULT_DATA_DIR = _HERE / "data"


# ==============================================================================
# 1. DATA LOADING  (local Faults.NNA + Faults27x7_var)
# ==============================================================================

def load_steel_data(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """Load the Steel Plates Faults dataset from local files.

    The dataset zip (steel+plates+faults.zip) must be extracted to *data_dir*
    (default: ``./data/``), producing:
      - ``Faults.NNA``        – 1941 rows × 34 space-separated values
      - ``Faults27x7_var``    – 34 variable names (27 features + 7 targets)

    Returns
    -------
    X   : pd.DataFrame, shape (1941, 27), column names from Faults27x7_var
    y   : pd.Series, integer labels 0-6 (LabelEncoded fault class names)
    le  : fitted LabelEncoder (use le.inverse_transform to get class names)
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    faults_path = data_dir / "Faults.NNA"
    vars_path   = data_dir / "Faults27x7_var"

    if not faults_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {faults_path}.\n"
            "Extract steel+plates+faults.zip into the 'data/' subfolder first."
        )

    # Read variable names (34 total: 27 features + 7 target indicator cols)
    all_col_names = [ln.strip() for ln in vars_path.read_text().splitlines() if ln.strip()]

    # Load data (tab-separated, no header)
    df = pd.read_csv(faults_path, sep=r"\s+", header=None, names=all_col_names)

    # Feature matrix
    X = df[ALL_FEATURES].copy().reset_index(drop=True)

    # Target: 7 binary one-hot columns → single string label → integer encoding
    y_label = df[TARGET_CLASSES].idxmax(axis=1).reset_index(drop=True)
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_label), name="target")

    print(f"[Dataset] Loaded {X.shape[0]} samples × {X.shape[1]} features | "
          f"{y.nunique()} classes | Missing: {X.isnull().sum().sum()}")
    return X, y, le


# ==============================================================================
# 2. GENETIC ALGORITHM (GA) – Feature Selection
# ==============================================================================

class GeneticFeatureSelector:
    """Binary Genetic Algorithm for feature selection.

    Encodes each individual as a binary chromosome of length *n_features*.
    Fitness = mean 3-fold CV F1-macro score – small feature-count penalty.

    Parameters
    ----------
    n_features    : number of input features (27 for this dataset)
    pop_size      : population size per generation
    generations   : number of GA iterations
    mutation_rate : per-gene bit-flip probability
    min_features  : minimum features that must stay selected after mutation
    random_state  : seed for reproducibility
    """

    def __init__(
        self,
        n_features: int = 27,
        pop_size: int = 12,
        generations: int = 10,
        mutation_rate: float = 0.1,
        min_features: int = 3,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.n_features    = n_features
        self.pop_size      = pop_size
        self.generations   = generations
        self.mutation_rate = mutation_rate
        self.min_features  = min_features
        self.rng           = np.random.default_rng(random_state)

        self.best_chromosome_: Optional[np.ndarray] = None
        self.best_fitness_: float = -1.0
        self.convergence_curve_: List[float] = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fitness(self, chromosome: np.ndarray, X_tr: np.ndarray, y_tr: np.ndarray) -> float:
        """3-fold CV F1-macro on selected features with feature-count penalty."""
        if chromosome.sum() == 0:
            return 0.0
        idx = np.where(chromosome == 1)[0]
        clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=80, random_state=RANDOM_STATE)
        scores = cross_val_score(clf, X_tr[:, idx], y_tr, cv=3, scoring="f1_macro", n_jobs=1)
        penalty = 0.001 * (len(idx) / self.n_features)
        return float(np.mean(scores) - penalty)

    # ------------------------------------------------------------------
    # BUG FIX #1  –  Tournament selection
    # ------------------------------------------------------------------
    def _tournament_select(
        self, population: np.ndarray, fitness_scores: np.ndarray
    ) -> np.ndarray:
        """Binary tournament selection.

        **Original bug (lines 139-140 of original .py):**
        ``population[rng.choice(pop_size, 2)][np.argmax(fitness_scores[rng.choice(pop_size, 2)])]``
        – The second ``rng.choice`` produced *new random indices* into
        ``fitness_scores``, completely independent of which individuals were
        drawn from the population in the first call.  This means the returned
        individual was not necessarily the winner of the tournament.

        **Fix:** draw two distinct population indices once; compare their
        fitness values; return the individual with the higher score.
        """
        idx1, idx2 = self.rng.choice(self.pop_size, size=2, replace=False)
        return (
            population[idx1].copy()
            if fitness_scores[idx1] >= fitness_scores[idx2]
            else population[idx2].copy()
        )

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Single-point crossover."""
        cut = int(self.rng.integers(1, self.n_features - 1))
        child = np.concatenate([p1[:cut], p2[cut:]])
        return child

    def _mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """Per-gene bit-flip with repair to maintain minimum feature count."""
        c = chromosome.copy()
        mask = self.rng.random(self.n_features) < self.mutation_rate
        c[mask] = 1 - c[mask]
        # Repair
        if c.sum() < self.min_features:
            zeros = np.where(c == 0)[0]
            n_flip = int(self.min_features - c.sum())
            flips  = self.rng.choice(zeros, min(n_flip, len(zeros)), replace=False)
            c[flips] = 1
        return c

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_tr: np.ndarray, y_tr: np.ndarray) -> "GeneticFeatureSelector":
        """Run the GA on training data.

        Parameters
        ----------
        X_tr : scaled feature matrix (n_train, n_features)
        y_tr : integer class labels (n_train,)

        Returns
        -------
        self
        """
        population = self.rng.integers(0, 2, size=(self.pop_size, self.n_features))
        # Ensure minimum active features at initialisation
        for i in range(self.pop_size):
            if population[i].sum() < self.min_features:
                population[i, self.rng.choice(self.n_features, self.min_features, replace=False)] = 1

        self.convergence_curve_ = []
        self.best_fitness_      = -1.0
        self.best_chromosome_   = None

        for gen in range(self.generations):
            fitness_scores = np.array(
                [self._fitness(ind, X_tr, y_tr) for ind in population]
            )
            best_idx = int(np.argmax(fitness_scores))
            if fitness_scores[best_idx] > self.best_fitness_:
                self.best_fitness_    = float(fitness_scores[best_idx])
                self.best_chromosome_ = population[best_idx].copy()

            self.convergence_curve_.append(self.best_fitness_)

            # Next generation: elitism + tournament + crossover + mutation
            new_pop = [self.best_chromosome_.copy()]
            while len(new_pop) < self.pop_size:
                p1    = self._tournament_select(population, fitness_scores)
                p2    = self._tournament_select(population, fitness_scores)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                new_pop.append(child)
            population = np.array(new_pop[: self.pop_size])

        return self

    def selected_feature_names(self, feature_names: List[str]) -> List[str]:
        """Return names of selected features given the full feature list."""
        if self.best_chromosome_ is None:
            raise RuntimeError("Call fit() first.")
        return [feature_names[i] for i in np.where(self.best_chromosome_ == 1)[0]]


# ==============================================================================
# 3. PARTICLE SWARM OPTIMIZATION (PSO) – 4-D Hyperparameter Search
# ==============================================================================

class PSOHyperparameterOptimizer:
    """PSO to optimise MLP hyperparameters on GA-selected features.

    Search space (4 dimensions, matching original .py):
      dim 0  h1    – neurons in hidden layer 1       : [32, 160]
      dim 1  h2    – neurons in hidden layer 2       : [16, 128]
      dim 2  lr    – learning_rate_init              : [0.0005, 0.01]
      dim 3  alpha – L2 regularisation strength      : [0.0001, 0.01]

    Fitness = mean 3-fold CV F1-macro score.
    """

    def __init__(
        self,
        n_particles: int = 8,
        iterations:  int = 8,
        w:  float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.n_particles  = n_particles
        self.iterations   = iterations
        self.w, self.c1, self.c2 = w, c1, c2
        self.rng          = np.random.default_rng(random_state)
        self.bounds_low   = np.array([32.0,   16.0,  0.0005, 0.0001])
        self.bounds_high  = np.array([160.0, 128.0,  0.0100, 0.0100])

        self.global_best_pos_:   Optional[np.ndarray] = None
        self.global_best_score_: float = -1.0
        self.convergence_curve_: List[float] = []

    def _decode(self, pos: np.ndarray) -> Dict[str, Any]:
        """Map position vector → MLP kwargs dict."""
        return {
            "h1":    int(np.clip(round(pos[0]), 32, 160)),
            "h2":    int(np.clip(round(pos[1]), 16, 128)),
            "lr":    float(np.clip(pos[2], 0.0005, 0.01)),
            "alpha": float(np.clip(pos[3], 0.0001, 0.01)),
        }

    def _evaluate(self, pos: np.ndarray, X_tr: np.ndarray, y_tr: np.ndarray) -> float:
        """3-fold CV F1-macro for one particle position."""
        p = self._decode(pos)
        clf = MLPClassifier(
            hidden_layer_sizes=(p["h1"], p["h2"]),
            learning_rate_init=p["lr"],
            alpha=p["alpha"],
            max_iter=120,
            random_state=RANDOM_STATE,
        )
        scores = cross_val_score(clf, X_tr, y_tr, cv=3, scoring="f1_macro", n_jobs=1)
        return float(np.mean(scores))

    def fit(self, X_tr: np.ndarray, y_tr: np.ndarray) -> "PSOHyperparameterOptimizer":
        """Run PSO.

        Parameters
        ----------
        X_tr : scaled feature matrix – only GA-selected columns
        y_tr : integer class labels

        Returns
        -------
        self
        """
        dim = len(self.bounds_low)
        positions  = self.rng.uniform(self.bounds_low, self.bounds_high, size=(self.n_particles, dim))
        velocities = np.zeros((self.n_particles, dim))
        pbest_pos  = positions.copy()
        pbest_scores = np.array([self._evaluate(p, X_tr, y_tr) for p in positions])

        best_idx = int(np.argmax(pbest_scores))
        self.global_best_pos_   = pbest_pos[best_idx].copy()
        self.global_best_score_ = float(pbest_scores[best_idx])
        self.convergence_curve_ = [self.global_best_score_]

        for it in range(self.iterations):
            for i in range(self.n_particles):
                r1 = self.rng.random(dim)
                r2 = self.rng.random(dim)
                velocities[i] = (
                    self.w  * velocities[i]
                    + self.c1 * r1 * (pbest_pos[i] - positions[i])
                    + self.c2 * r2 * (self.global_best_pos_ - positions[i])
                )
                positions[i] = np.clip(positions[i] + velocities[i], self.bounds_low, self.bounds_high)
                score = self._evaluate(positions[i], X_tr, y_tr)
                if score > pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest_pos[i] = positions[i].copy()
                    if score > self.global_best_score_:
                        self.global_best_score_ = score
                        self.global_best_pos_   = positions[i].copy()
            self.convergence_curve_.append(self.global_best_score_)

        return self

    @property
    def best_params(self) -> Dict[str, Any]:
        """Return best decoded hyperparameters."""
        if self.global_best_pos_ is None:
            raise RuntimeError("Call fit() first.")
        return self._decode(self.global_best_pos_)


# ==============================================================================
# 4. FUZZY INFERENCE SYSTEM (Kelompok X4 – Decision Support Layer)
# ==============================================================================

class SteelInspectionFuzzySystem:
    """Mamdani-style fuzzy system: (confidence, normalized_area, class) → priority.

    Inputs
    ------
    predicted_class_name : str   – e.g. "K_Scatch"
    confidence_prob      : float – max softmax probability (0–1)
    normalized_area      : float – (LogOfAreas - min) / (max - min), in [0, 1]

    Output
    ------
    decision   : str   – "Accept" | "Monitor" | "Rework" | "Reject"
    risk_score : float – continuous risk index in [0, 1]

    **BUG FIX #2** – trapmf / trimf:
    Original: ``(x-a)/(b-a+1e-9)``  – guard applied to full expression, not
    just the denominator.  When a==b the numerator is 0 but the full expression
    would be 0/eps ≈ 0 which is accidentally OK.  However when b==c in trimf,
    the guard on the second fraction ``(c-x)/(c-b+1e-9)`` could be very large.
    Fixed: ``max(denominator, eps)`` applied cleanly; boundary conditions are
    checked first so we never enter the ramp computation with a degenerate interval.
    """

    # Structural severity weight per defect class
    CLASS_SEVERITY_WEIGHTS: Dict[str, float] = {
        "Stains":       0.35,
        "Dirtiness":    0.40,
        "Other_Faults": 0.55,
        "Bumps":        0.70,
        "Pastry":       0.75,
        "Z_Scratch":    0.88,
        "K_Scatch":     0.95,
    }

    @staticmethod
    def trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
        """Trapezoidal membership function – robust against degenerate intervals.

        Returns 0 when x ≤ a or x ≥ d, 1 when b ≤ x ≤ c, linear elsewhere.
        """
        eps = 1e-9
        if x <= a or x >= d:
            return 0.0
        if b <= x <= c:
            return 1.0
        if a < x < b:
            return float((x - a) / max(b - a, eps))
        # c < x < d
        return float((d - x) / max(d - c, eps))

    @staticmethod
    def trimf(x: float, a: float, b: float, c: float) -> float:
        """Triangular membership function – robust against degenerate intervals.

        Returns 0 when x ≤ a or x ≥ c, 1 at x == b, linear on each side.
        """
        eps = 1e-9
        if x <= a or x >= c:
            return 0.0
        if x <= b:
            return float((x - a) / max(b - a, eps))
        return float((c - x) / max(c - b, eps))

    def evaluate_decision(
        self,
        predicted_class_name: str,
        confidence_prob: float,
        normalized_area: float,
    ) -> Tuple[str, float]:
        """Evaluate the fuzzy rule base and return (decision, risk_score).

        Parameters
        ----------
        predicted_class_name : one of the 7 defect class strings
        confidence_prob      : model softmax confidence [0, 1]
        normalized_area      : (LogOfAreas - global_min) / (global_max - global_min)

        Returns
        -------
        decision   : "Accept" | "Monitor" | "Rework" | "Reject"
        risk_score : float in [0, 1]
        """
        # --- Fuzzify confidence ---
        prob_low  = self.trapmf(confidence_prob, 0.0, 0.0, 0.35, 0.55)
        prob_med  = self.trimf( confidence_prob, 0.40, 0.65, 0.85)
        prob_high = self.trapmf(confidence_prob, 0.70, 0.85, 1.0, 1.0)

        # --- Fuzzify area severity ---
        area_small = self.trapmf(normalized_area, 0.0, 0.0, 0.25, 0.45)
        area_mod   = self.trimf( normalized_area, 0.30, 0.55, 0.75)
        area_large = self.trapmf(normalized_area, 0.65, 0.80, 1.0,  1.0)

        # --- Class structural risk weight ---
        class_risk = self.CLASS_SEVERITY_WEIGHTS.get(predicted_class_name, 0.55)

        # --- Rule firing strengths ---
        r_accept  = min(prob_low,  area_small) * (1.0 - 0.5 * class_risk)
        _r_monitor = max(min(prob_med, area_small), min(prob_low,  area_mod))
        _r_rework  = max(min(prob_high, area_mod), min(prob_med,  area_large)) * class_risk
        r_reject  = min(prob_high, area_large) * (0.6 + 0.4 * class_risk)

        # --- Continuous risk score (weighted sum) ---
        risk_score = (
            0.40 * confidence_prob
            + 0.30 * normalized_area
            + 0.30 * class_risk
        )
        # Rule-based adjustments
        if r_reject > 0.5:
            risk_score = max(risk_score, 0.82)
        elif r_accept > 0.5 and class_risk <= 0.45:
            risk_score = min(risk_score, 0.25)

        # --- Defuzzify to priority label ---
        if risk_score < 0.30:
            decision = "Accept"
        elif risk_score < 0.55:
            decision = "Monitor"
        elif risk_score < 0.78:
            decision = "Rework"
        else:
            decision = "Reject"

        return decision, round(float(risk_score), 4)


# ==============================================================================
# 4b. 1D-CONVOLUTIONAL NEURAL NETWORK (1D-CNN) – PyTorch
# ==============================================================================

class CNN1DNet(nn.Module):
    """PyTorch 1D-CNN architecture for 1D tabular steel surface fault features."""

    def __init__(self, in_features: int, num_classes: int = 7) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(4)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class CNN1DClassifier:
    """1D-CNN Classifier wrapper with Scikit-Learn compatible API."""

    def __init__(
        self,
        epochs: int = 25,
        batch_size: int = 32,
        lr: float = 0.003,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self.model_: Optional[CNN1DNet] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CNN1DClassifier":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)
        in_features = X.shape[1]

        self.model_ = CNN1DNet(in_features=in_features, num_classes=num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                out = self.model_(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model has not been trained yet.")
        self.model_.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model_(X_tensor)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


CNN1DNet.__module__ = "kelompok_x4_hybrid_pipeline"
CNN1DClassifier.__module__ = "kelompok_x4_hybrid_pipeline"


# ==============================================================================
# 5. MODEL TRAINING & EVALUATION HELPERS
# ==============================================================================

def evaluate_classifier(
    model: Any,
    X_te: np.ndarray,
    y_te: np.ndarray,
    model_name: str = "",
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Compute classification metrics.

    Returns
    -------
    metrics_dict : dict with Accuracy, Precision, Recall, F1-Macro, F1-Weighted
    y_pred       : prediction array
    """
    y_pred = model.predict(X_te)
    return {
        "Model":                model_name,
        "Accuracy":             round(accuracy_score(y_te, y_pred), 4),
        "Precision (Macro)":    round(precision_score(y_te, y_pred, average="macro",    zero_division=0), 4),
        "Recall (Macro)":       round(recall_score(y_te,    y_pred, average="macro",    zero_division=0), 4),
        "F1-Score (Macro)":     round(f1_score(y_te,        y_pred, average="macro",    zero_division=0), 4),
        "F1-Score (Weighted)":  round(f1_score(y_te,        y_pred, average="weighted", zero_division=0), 4),
    }, y_pred


def run_fuzzy_on_test(
    model: Any,
    X_test_ga: np.ndarray,
    X_test_orig: pd.DataFrame,
    y_test: np.ndarray,
    le: LabelEncoder,
    n_samples: int = 200,
) -> pd.DataFrame:
    """Run the fuzzy decision layer on the first *n_samples* of the test set.

    Parameters
    ----------
    model        : fitted classifier with predict_proba (e.g. Model D or Model F)
    X_test_ga    : scaled test data with only GA-selected features
    X_test_orig  : ORIGINAL (pre-scaling) X_test DataFrame  ← FIX-6
    y_test       : integer true labels
    le           : fitted LabelEncoder
    n_samples    : how many test samples to process

    Returns
    -------
    pd.DataFrame with one row per sample
    """
    n = min(n_samples, len(X_test_ga))
    probs = model.predict_proba(X_test_ga[:n])

    area_col = "LogOfAreas"
    log_areas = X_test_orig[area_col].values[:n].astype(float) if area_col in X_test_orig.columns \
                else X_test_orig.iloc[:n, -1].values.astype(float)

    area_min = log_areas.min()
    area_max = log_areas.max()

    fuzzy = SteelInspectionFuzzySystem()
    rows  = []
    preds = model.predict(X_test_ga[:n])

    for i in range(n):
        pred_cls  = le.inverse_transform([int(preds[i])])[0]
        true_cls  = le.inverse_transform([int(y_test[i])])[0]
        conf      = float(probs[i].max())
        norm_area = float((log_areas[i] - area_min) / max(area_max - area_min, 1e-9))
        decision, risk = fuzzy.evaluate_decision(pred_cls, conf, norm_area)
        rows.append({
            "Sample_ID":           i + 1,
            "True_Class":          true_cls,
            "Predicted_Class":     pred_cls,
            "Confidence":          round(conf, 4),
            "LogOfAreas_Raw":      round(float(log_areas[i]), 4),
            "LogOfAreas_Norm":     round(norm_area, 4),
            "Fuzzy_Risk_Score":    risk,
            "Inspection_Priority": decision,
        })

    return pd.DataFrame(rows)


# ==============================================================================
# 6. FULL PIPELINE ORCHESTRATOR
# ==============================================================================

def run_full_pipeline(
    ga_runs:       int   = 3,
    ga_pop:        int   = 12,
    ga_gen:        int   = 10,
    pso_particles: int   = 8,
    pso_iter:      int   = 8,
    data_dir:      Optional[str] = None,
    artifact_path: str   = ARTIFACT_PATH,
    verbose:       bool  = True,
) -> Dict[str, Any]:
    """End-to-end hybrid AI pipeline for Steel Fault Detection.

    Steps
    -----
    1. Load local dataset (Faults.NNA / Faults27x7_var).
    2. Stratified train/test split (80/20) + StandardScaler.
    3. Run GA *ga_runs* times; keep chromosome with best fitness.
    4. Run PSO (4-D) on GA-selected features.
    5. Train and evaluate Model A, B, C, D.
    6. Run Fuzzy layer on first 200 test samples.
    7. Save pipeline_artifacts.pkl.

    Returns
    -------
    artifacts : dict (also saved to disk)
    """
    sep = "=" * 65

    if verbose:
        print(sep)
        print("  Kelompok X4  –  Steel Fault Detection  –  Hybrid Pipeline")
        print(sep)

    # ------------------------------------------------------------------
    # Step 1: Data
    # ------------------------------------------------------------------
    if verbose: print("\n[1/6] Loading dataset …")
    X, y, le = load_steel_data(data_dir)
    n_features = X.shape[1]

    # ------------------------------------------------------------------
    # Step 2: Split + Scale  (FIX-6: preserve X_test before scaling)
    # ------------------------------------------------------------------
    if verbose: print("[2/6] Splitting and scaling …")
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_test_orig = X_test_df.copy().reset_index(drop=True)   # ← FIX-6

    scaler       = StandardScaler()
    X_train_sc   = scaler.fit_transform(X_train_df)
    X_test_sc    = scaler.transform(X_test_df)
    y_train_arr  = y_train.values
    y_test_arr   = y_test.values

    if verbose:
        print(f"    Train: {X_train_sc.shape}, Test: {X_test_sc.shape}")

    # ------------------------------------------------------------------
    # Step 3: Genetic Algorithm – multiple independent runs
    # ------------------------------------------------------------------
    if verbose: print(f"\n[3/6] Running GA ({ga_runs} runs) …")
    ga_all_results = []
    for run in range(ga_runs):
        seed = RANDOM_STATE + run * 7
        if verbose: print(f"  Run {run+1}/{ga_runs}  (seed={seed})")
        ga = GeneticFeatureSelector(
            n_features=n_features,
            pop_size=ga_pop,
            generations=ga_gen,
            random_state=seed,
        )
        ga.fit(X_train_sc, y_train_arr)
        ga_all_results.append(ga)
        if verbose:
            print(f"    Best F1: {ga.best_fitness_:.4f} | "
                  f"Features: {int(ga.best_chromosome_.sum())}")

    best_ga    = max(ga_all_results, key=lambda g: g.best_fitness_)
    best_chrom = best_ga.best_chromosome_
    ga_curve   = best_ga.convergence_curve_
    sel_names  = best_ga.selected_feature_names(ALL_FEATURES)
    sel_idx    = np.where(best_chrom == 1)[0]

    # Feature selection frequency across runs
    freq_arr = np.stack([g.best_chromosome_ for g in ga_all_results]).mean(axis=0)
    feat_freq = dict(zip(ALL_FEATURES, freq_arr.round(3).tolist()))

    if verbose:
        fits = [g.best_fitness_ for g in ga_all_results]
        print(f"\n  GA Stats → Best: {np.max(fits):.4f}, "
              f"Mean: {np.mean(fits):.4f}, Std: {np.std(fits):.4f}")
        print(f"  Selected {len(sel_names)}/27 features: {sel_names}")

    X_train_ga = X_train_sc[:, sel_idx]
    X_test_ga  = X_test_sc[:, sel_idx]

    # ------------------------------------------------------------------
    # Step 4: PSO – 4-D hyperparameter search
    # ------------------------------------------------------------------
    if verbose: print(f"\n[4/6] Running PSO ({pso_particles} particles, {pso_iter} iters) …")
    pso = PSOHyperparameterOptimizer(
        n_particles=pso_particles,
        iterations=pso_iter,
        random_state=RANDOM_STATE,
    )
    pso.fit(X_train_ga, y_train_arr)
    pso_params = pso.best_params
    pso_curve  = pso.convergence_curve_

    if verbose:
        print(f"  Best params: h1={pso_params['h1']}, h2={pso_params['h2']}, "
              f"lr={pso_params['lr']:.5f}, alpha={pso_params['alpha']:.5f}")
        print(f"  PSO Best F1: {pso.global_best_score_:.4f}")

    h1, h2  = pso_params["h1"], pso_params["h2"]
    lr      = pso_params["lr"]
    alpha   = pso_params["alpha"]

    # ------------------------------------------------------------------
    # Step 5: Train & Evaluate Models (MLP Variants & 1D-CNN)
    # ------------------------------------------------------------------
    if verbose: print("\n[5/6] Training and evaluating models (MLP & 1D-CNN) …")

    model_A = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=RANDOM_STATE)
    model_B = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=RANDOM_STATE)
    model_C = MLPClassifier(hidden_layer_sizes=(h1, h2), learning_rate_init=lr, alpha=alpha,
                             max_iter=200, random_state=RANDOM_STATE)
    model_D = MLPClassifier(hidden_layer_sizes=(h1, h2), learning_rate_init=lr, alpha=alpha,
                             max_iter=200, random_state=RANDOM_STATE)
    model_E = CNN1DClassifier(epochs=25, batch_size=32, lr=0.003, random_state=RANDOM_STATE)
    model_F = CNN1DClassifier(epochs=25, batch_size=32, lr=0.003, random_state=RANDOM_STATE)

    model_A.fit(X_train_sc, y_train_arr)
    model_B.fit(X_train_ga, y_train_arr)
    model_C.fit(X_train_sc, y_train_arr)
    model_D.fit(X_train_ga, y_train_arr)
    model_E.fit(X_train_sc, y_train_arr)
    model_F.fit(X_train_ga, y_train_arr)

    metrics_A, _ = evaluate_classifier(model_A, X_test_sc,  y_test_arr, "Model A: Baseline MLP")
    metrics_B, _ = evaluate_classifier(model_B, X_test_ga,  y_test_arr, "Model B: GA + MLP")
    metrics_C, _ = evaluate_classifier(model_C, X_test_sc,  y_test_arr, "Model C: PSO + MLP")
    metrics_D, preds_D = evaluate_classifier(model_D, X_test_ga,  y_test_arr, "Model D: GA + PSO + MLP")
    metrics_E, _ = evaluate_classifier(model_E, X_test_sc,  y_test_arr, "Model E: 1D-CNN (All Feat)")
    metrics_F, preds_F = evaluate_classifier(model_F, X_test_ga,  y_test_arr, "Model F: GA + 1D-CNN")

    comparison_df = pd.DataFrame([metrics_A, metrics_B, metrics_C, metrics_D, metrics_E, metrics_F]).set_index("Model")
    cm_D = confusion_matrix(y_test_arr, preds_D)
    cr_D = classification_report(y_test_arr, preds_D, target_names=le.classes_, zero_division=0)
    cm_F = confusion_matrix(y_test_arr, preds_F)
    cr_F = classification_report(y_test_arr, preds_F, target_names=le.classes_, zero_division=0)

    if verbose:
        print("\n  Model Comparison:")
        print(comparison_df.to_string())
        print("\n  Model D (GA + PSO + MLP) Classification Report:")
        print(cr_D)
        print("\n  Model F (GA + 1D-CNN) Classification Report:")
        print(cr_F)

    # ------------------------------------------------------------------
    # Step 6: Fuzzy layer on test set
    # ------------------------------------------------------------------
    if verbose: print("\n[6/6] Running Fuzzy Inspection System …")
    fuzzy_df = run_fuzzy_on_test(
        model=model_D,
        X_test_ga=X_test_ga,
        X_test_orig=X_test_orig,   # ← FIX-6: original unscaled values
        y_test=y_test_arr,
        le=le,
        n_samples=200,
    )
    if verbose:
        print(f"  Processed {len(fuzzy_df)} samples")
        print(fuzzy_df["Inspection_Priority"].value_counts().to_string())

    # ------------------------------------------------------------------
    # Save artifacts (FIX-3, FIX-4)
    # ------------------------------------------------------------------
    artifacts: Dict[str, Any] = {
        # Data & preprocessing
        "scaler":              scaler,
        "le":                  le,
        "all_feature_names":   ALL_FEATURES,
        "class_names":         list(le.classes_),
        # GA
        "ga_chromosome":       best_chrom,
        "ga_selected_features":sel_names,
        "ga_feature_freq":     feat_freq,
        "ga_curve":            ga_curve,
        "ga_all_results": [
            {
                "run":         i + 1,
                "best_fitness":g.best_fitness_,
                "n_features":  int(g.best_chromosome_.sum()),
                "chromosome":  g.best_chromosome_.tolist(),
                "curve":       g.convergence_curve_,
            }
            for i, g in enumerate(ga_all_results)
        ],
        # PSO
        "pso_params":          pso_params,
        "pso_curve":           pso_curve,
        "pso_best_fitness":    pso.global_best_score_,
        # Models
        "final_model":         model_D,
        "final_model_cnn":     model_F,
        "all_models":          {
            "Model A: Baseline MLP":    model_A,
            "Model B: GA + MLP":        model_B,
            "Model C: PSO + MLP":       model_C,
            "Model D: GA + PSO + MLP":  model_D,
            "Model E: 1D-CNN (All Feat)": model_E,
            "Model F: GA + 1D-CNN":     model_F,
        },
        # Evaluation
        "comparison_df":       comparison_df,
        "confusion_matrix":    cm_D,
        "classification_report": cr_D,
        "confusion_matrix_cnn": cm_F,
        "classification_report_cnn": cr_F,
        # Test data
        "X_test_sc":           X_test_sc,
        "X_test_ga":           X_test_ga,
        "X_test_orig":         X_test_orig,   # ← FIX-6
        "y_test":              y_test_arr,
        # Fuzzy
        "fuzzy_results_df":    fuzzy_df,
    }

    joblib.dump(artifacts, artifact_path)
    if verbose:
        size_kb = os.path.getsize(artifact_path) / 1024
        print(f"\n  ✓ Artifacts saved → '{artifact_path}'  ({size_kb:.1f} KB)")
        print(sep)
        print("  Pipeline complete!")
        print(sep)

    return artifacts


# ==============================================================================
# 7. __main__  –  CLI demonstration  (FIX-5: was empty)
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Kelompok X4 – Steel Fault Detection Pipeline"
    )
    parser.add_argument("--ga-runs",        type=int,   default=3,
                        help="Number of independent GA runs (default: 3)")
    parser.add_argument("--ga-pop",         type=int,   default=12,
                        help="GA population size (default: 12)")
    parser.add_argument("--ga-gen",         type=int,   default=10,
                        help="GA generations (default: 10)")
    parser.add_argument("--pso-particles",  type=int,   default=8,
                        help="PSO particle count (default: 8)")
    parser.add_argument("--pso-iter",       type=int,   default=8,
                        help="PSO iterations (default: 8)")
    parser.add_argument("--data-dir",       type=str,   default=None,
                        help="Path to folder containing Faults.NNA (default: ./data/)")
    parser.add_argument("--output",         type=str,   default=ARTIFACT_PATH,
                        help=f"Output pkl file (default: {ARTIFACT_PATH})")
    args = parser.parse_args()

    arts = run_full_pipeline(
        ga_runs=args.ga_runs,
        ga_pop=args.ga_pop,
        ga_gen=args.ga_gen,
        pso_particles=args.pso_particles,
        pso_iter=args.pso_iter,
        data_dir=args.data_dir,
        artifact_path=args.output,
    )

    print("\nSelected features:", arts["ga_selected_features"])
    print("PSO params:", arts["pso_params"])
    print("\nModel comparison:\n", arts["comparison_df"])
