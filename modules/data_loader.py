"""
data_loader.py
==============
Loads the Steel Plates Faults dataset from local files.

Dataset : UCI Steel Plates Faults (ID 198)
Files   : data/Faults.NNA  +  data/Faults27x7_var
Samples : 1941 × 27 features, 7 target classes
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

_HERE = Path(__file__).parent.parent
DEFAULT_DATA_DIR = _HERE / "data"


def load_steel_data(
    data_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """Load Steel Plates Faults dataset from local files.

    Parameters
    ----------
    data_dir : path to folder containing Faults.NNA and Faults27x7_var.
               Defaults to <project_root>/data/

    Returns
    -------
    X   : pd.DataFrame, shape (1941, 27)
    y   : pd.Series of integer labels 0-6
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

    all_col_names = [ln.strip() for ln in vars_path.read_text().splitlines() if ln.strip()]
    df = pd.read_csv(faults_path, sep=r"\s+", header=None, names=all_col_names)

    X = df[ALL_FEATURES].copy().reset_index(drop=True)
    y_label = df[TARGET_CLASSES].idxmax(axis=1).reset_index(drop=True)
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_label), name="target")

    print(f"[Dataset] Loaded {X.shape[0]} samples x {X.shape[1]} features | "
          f"{y.nunique()} classes | Missing: {X.isnull().sum().sum()}")
    return X, y, le
