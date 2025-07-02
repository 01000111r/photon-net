"""
Global configuration variables for the photonic classifier model.
"""
import os
from pathlib import Path
import platform

# The total number of optimization steps to perform during training.
num_steps: int = 50

# The frequency at which data is re-uploaded into the circuit.
# A new data layer is introduced every `reupload_freq` layers.
reupload_freq: int = 3

# The number of modes in the photonic circuit.
num_modes_circ: int = 6

# The user only needs to modify this FEATURE_SIZE.
FEATURE_SIZE = 3







# Get the project root directory. Assuming p_pack is a subfolder in your codebase.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directory for mnist_pca data folder (should be in the codebase).
MNIST_PCA_DIR = PROJECT_ROOT / "mnist_pca"

def get_mnist_csv_filepath(split: str) -> str:
    """
    Returns the file path for the MNIST CSV file for the given split.

    Args:
        split (str): 'train' or 'test'.

    Returns:
        str: Full platform-independent file path.
    """
    # Construct filename e.g. "mnist_3-5_3d_train.csv"
    fname = f"mnist_3-5_{FEATURE_SIZE}d_{split}.csv"
    return str(MNIST_PCA_DIR / fname)










