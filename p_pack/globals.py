"""
Global configuration variables for the photonic classifier model.
"""
import os
from pathlib import Path
import platform
import sys
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import jax
import jax.numpy as jnp
from jax import grad
from numpy.random import default_rng
from functools import partial
import time
import pandas as pd
from itertools import combinations
import itertools
from jax.scipy.special import factorial
import numpy as np
#from scipy.special import factorial

from thewalrus import perm


# The total number of optimization steps to perform during training.
num_steps: int = 50

#training rate
training_rate : float =  1e-2

# The frequency at which data is re-uploaded into the circuit.
# A new data layer is introduced every `reupload_freq` layers.
reupload_freq: int = 3

# The number of modes in the photonic circuit.
num_modes_circ: int = 6

# The user only needs to modify this FEATURE_SIZE.
num_features = 3







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
    fname = f"mnist_3-5_{num_features}d_{split}.csv"
    return str(MNIST_PCA_DIR / fname)


def load_and_split_data(num_features):

    # Load MNIST data for handwritten digits '3' and '5'
    # data_set = (x,y)
    # 'x' rows (image number), number of _images
    # 'y' columns (pixel/feature number): first y-1 columns correspond to pixel features - last column is image label 
    # lables: +1 = '3', -1 = '5'

    # load training set
    file_path = get_mnist_csv_filepath("train")
    data_train = pd.read_csv(file_path)
    data_train = jnp.array(data_train)

    #laod test set
    file_path_1 = get_mnist_csv_filepath("test")
    data_test = pd.read_csv(file_path_1)
    data_test = jnp.array(data_test)

    #splitting training set + test set into features and labels
    num_features = data_train.shape[1] -1 
    train_set = data_train[:,:num_features]
    train_labels = data_train[:,num_features]
    test_set = data_test[:,:num_features]
    test_labels = data_test[:,num_features]

    return train_set, train_labels, test_set, test_labels











