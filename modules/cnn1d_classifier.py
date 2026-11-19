"""
cnn1d_classifier.py
===================
1D Convolutional Neural Network (PyTorch) for tabular steel fault classification.

Architecture:
  Input  : (batch, n_features)       -- 2D input auto-unsqueezed to (batch, 1, n_features)
  Conv1  : Conv1d(1->32, k=3, pad=1) -> BatchNorm1d(32) -> ReLU
  Conv2  : Conv1d(32->64, k=3, pad=1) -> BatchNorm1d(64) -> ReLU
  Pool   : AdaptiveAvgPool1d(4)      -- output: (batch, 64, 4)
  Flatten: (batch, 256)
  FC     : Dropout(0.3) -> Linear(256->64) -> ReLU -> Dropout(0.2) -> Linear(64->n_classes)
  Output : logits (batch, n_classes)

Training : Adam optimizer, CrossEntropyLoss, weight_decay=1e-4
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .data_loader import RANDOM_STATE


class CNN1DNet(nn.Module):
    """PyTorch 1D-CNN module for tabular feature classification."""

    def __init__(self, in_features: int, num_classes: int = 7) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(32)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(4)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Auto-handles (batch, features) or (batch, 1, features)."""
        if x.dim() == 2:
            x = x.unsqueeze(1)                        # -> (batch, 1, features)
        x = self.relu(self.bn1(self.conv1(x)))        # -> (batch, 32, features)
        x = self.relu(self.bn2(self.conv2(x)))        # -> (batch, 64, features)
        x = self.adaptive_pool(x)                     # -> (batch, 64, 4)
        x = torch.flatten(x, 1)                       # -> (batch, 256)
        return self.fc(x)                             # -> (batch, n_classes)


class CNN1DClassifier:
    """Scikit-Learn compatible wrapper for CNN1DNet.

    Parameters
    ----------
    epochs       : number of training epochs
    batch_size   : mini-batch size for DataLoader
    lr           : Adam optimizer learning rate
    random_state : seed for PyTorch and NumPy
    """

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
        """Train the 1D-CNN on feature matrix X and labels y."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.classes_   = np.unique(y)
        num_classes     = len(self.classes_)
        in_features     = X.shape[1]

        self.model_ = CNN1DNet(in_features=in_features, num_classes=num_classes)
        criterion   = nn.CrossEntropyLoss()
        optimizer   = optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)

        X_t    = torch.tensor(X, dtype=torch.float32)
        y_t    = torch.tensor(y, dtype=torch.long)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for _ in range(self.epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(self.model_(bx), by)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax probabilities, shape (n_samples, n_classes)."""
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        self.model_.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            proba = torch.softmax(self.model_(X_t), dim=1).cpu().numpy()
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices, shape (n_samples,)."""
        return np.argmax(self.predict_proba(X), axis=1)


# Critical: fix __module__ for joblib pickle compatibility when loaded by Streamlit
CNN1DNet.__module__        = "kelompok_x4_hybrid_pipeline"
CNN1DClassifier.__module__ = "kelompok_x4_hybrid_pipeline"
