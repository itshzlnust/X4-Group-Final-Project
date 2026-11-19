"""
ga_feature_selector.py
======================
Binary Genetic Algorithm for feature selection.

Chromosome encoding : binary vector of length n_features (27)
Fitness function    : mean 3-fold CV F1-macro - feature-count penalty
Selection           : binary tournament
Crossover           : single-point
Mutation            : per-gene bit-flip with minimum-feature repair
Elitism             : best individual always copied to next generation
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

from .data_loader import RANDOM_STATE


class GeneticFeatureSelector:
    """Binary Genetic Algorithm for feature selection.

    Parameters
    ----------
    n_features    : number of input features (27 for steel dataset)
    pop_size      : population size per generation
    generations   : number of GA iterations
    mutation_rate : per-gene bit-flip probability
    min_features  : minimum features that must remain selected after mutation
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
        """Fitness = F1_macro_CV - feature_count_penalty.

        fitness(c) = mean_CV_F1_macro(c) - 0.001 * (|c| / n_features)
        """
        if chromosome.sum() == 0:
            return 0.0
        idx = np.where(chromosome == 1)[0]
        clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=80, random_state=RANDOM_STATE)
        scores = cross_val_score(clf, X_tr[:, idx], y_tr, cv=3, scoring="f1_macro", n_jobs=1)
        penalty = 0.001 * (len(idx) / self.n_features)
        return float(np.mean(scores) - penalty)

    def _tournament_select(
        self, population: np.ndarray, fitness_scores: np.ndarray
    ) -> np.ndarray:
        """Binary tournament: draw 2 random individuals, return the fitter one."""
        idx1, idx2 = self.rng.choice(self.pop_size, size=2, replace=False)
        return (
            population[idx1].copy()
            if fitness_scores[idx1] >= fitness_scores[idx2]
            else population[idx2].copy()
        )

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Single-point crossover at a uniformly random cut point."""
        cut = int(self.rng.integers(1, self.n_features - 1))
        return np.concatenate([p1[:cut], p2[cut:]])

    def _mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """Per-gene bit-flip with repair to maintain minimum feature count."""
        c = chromosome.copy()
        mask = self.rng.random(self.n_features) < self.mutation_rate
        c[mask] = 1 - c[mask]
        # Repair: ensure at least min_features are active
        if c.sum() < self.min_features:
            zeros  = np.where(c == 0)[0]
            n_flip = int(self.min_features - c.sum())
            flips  = self.rng.choice(zeros, min(n_flip, len(zeros)), replace=False)
            c[flips] = 1
        return c

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_tr: np.ndarray, y_tr: np.ndarray) -> "GeneticFeatureSelector":
        """Run GA on training data and store the best chromosome.

        Parameters
        ----------
        X_tr : scaled feature matrix, shape (n_train, n_features)
        y_tr : integer class labels, shape (n_train,)

        Returns
        -------
        self
        """
        # Initialise random binary population
        population = self.rng.integers(0, 2, size=(self.pop_size, self.n_features))
        for i in range(self.pop_size):
            if population[i].sum() < self.min_features:
                population[i, self.rng.choice(self.n_features, self.min_features, replace=False)] = 1

        self.convergence_curve_ = []
        self.best_fitness_      = -1.0
        self.best_chromosome_   = None

        for _ in range(self.generations):
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
        """Return names of features selected by the best chromosome."""
        if self.best_chromosome_ is None:
            raise RuntimeError("Call fit() first.")
        return [feature_names[i] for i in np.where(self.best_chromosome_ == 1)[0]]
