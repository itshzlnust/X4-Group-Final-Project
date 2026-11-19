"""
pso_optimizer.py
================
Particle Swarm Optimization for MLP hyperparameter tuning.

Search space (4-dimensional):
  dim 0  h1    - neurons hidden layer 1   : [32, 160]
  dim 1  h2    - neurons hidden layer 2   : [16, 128]
  dim 2  lr    - learning rate init       : [0.0005, 0.01]
  dim 3  alpha - L2 regularisation        : [0.0001, 0.01]

Velocity update equation:
  v_i(t+1) = w * v_i(t)
            + c1 * r1 * (pbest_i - x_i(t))
            + c2 * r2 * (gbest   - x_i(t))
  x_i(t+1) = clip(x_i(t) + v_i(t+1), bounds_low, bounds_high)

Fitness: mean 3-fold CV F1-macro score
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

from .data_loader import RANDOM_STATE


class PSOHyperparameterOptimizer:
    """PSO to find optimal MLP hyperparameters on GA-selected features.

    Parameters
    ----------
    n_particles  : swarm size
    iterations   : number of PSO iterations
    w            : inertia weight (controls momentum)
    c1           : cognitive coefficient (pull toward personal best)
    c2           : social coefficient (pull toward global best)
    random_state : seed for reproducibility
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
        """Map continuous position vector to MLP hyperparameter dict."""
        return {
            "h1":    int(np.clip(round(pos[0]), 32, 160)),
            "h2":    int(np.clip(round(pos[1]), 16, 128)),
            "lr":    float(np.clip(pos[2], 0.0005, 0.01)),
            "alpha": float(np.clip(pos[3], 0.0001, 0.01)),
        }

    def _evaluate(self, pos: np.ndarray, X_tr: np.ndarray, y_tr: np.ndarray) -> float:
        """3-fold CV F1-macro fitness for one particle position."""
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
        """Run PSO and store the global best hyperparameters.

        Parameters
        ----------
        X_tr : scaled feature matrix — only GA-selected columns
        y_tr : integer class labels

        Returns
        -------
        self
        """
        dim = len(self.bounds_low)
        positions    = self.rng.uniform(self.bounds_low, self.bounds_high, size=(self.n_particles, dim))
        velocities   = np.zeros((self.n_particles, dim))
        pbest_pos    = positions.copy()
        pbest_scores = np.array([self._evaluate(p, X_tr, y_tr) for p in positions])

        best_idx = int(np.argmax(pbest_scores))
        self.global_best_pos_   = pbest_pos[best_idx].copy()
        self.global_best_score_ = float(pbest_scores[best_idx])
        self.convergence_curve_ = [self.global_best_score_]

        for _ in range(self.iterations):
            for i in range(self.n_particles):
                r1 = self.rng.random(dim)
                r2 = self.rng.random(dim)
                velocities[i] = (
                    self.w  * velocities[i]
                    + self.c1 * r1 * (pbest_pos[i] - positions[i])
                    + self.c2 * r2 * (self.global_best_pos_ - positions[i])
                )
                positions[i] = np.clip(
                    positions[i] + velocities[i], self.bounds_low, self.bounds_high
                )
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
        """Return best decoded hyperparameters as dict {h1, h2, lr, alpha}."""
        if self.global_best_pos_ is None:
            raise RuntimeError("Call fit() first.")
        return self._decode(self.global_best_pos_)
