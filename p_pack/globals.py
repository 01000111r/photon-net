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
num_steps: int = 10

#training rate
training_rate : float =  0.1

# The frequency at which data is re-uploaded into the circuit.
# A new data layer is introduced every `reupload_freq` layers.
reupload_freq: int = 3

# The number of modes in the photonic circuit.
num_modes_circ: int = 10

# The number of layers in the photonic circuit.
depth: int = 10

# The user only needs to modify this FEATURE_SIZE.
num_features = 3

# probability of sucess for each mode /source looss
p_suc_inputs = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]

input_positions = '3'

# input config - tuple ( input_positions, p_suc_inputs)
input_config = None

#Aim for number of photons
aim = 3

# 0 to not discard, 1 to discard 
discard = 1

# Comparison operator used when deciding whether to discard
# training updates based on the number of photons measured.
# Possible values: '!=', '<', '<=', '>', '>=', '=='
discard_condition = '!='

# Optional range for photon counts when using discard_condition='range'.
# Should be a tuple (min_value, max_value). Set to None to disable.
discard_range = None

#switch for the trainable small photon loss scale, 0 - for no scaling, 1 - for scaling
loss_function = 0

# Batch processing configuration
# 'full'  : use entire dataset each update
# 'mini'  : use mini-batches of size ``mini_batch_size``
# 'single': process one sample at a time
batch_mode: str = 'full'
mini_batch_size: int = 16


max_photons = 3

def input_config_maker(input : str, num_modes: int, p_suc_inputs:list) -> tuple:
    """
    Converts a input string into an input coinfigurations
    
    Args:
        input_string (str): 'full' or 'n' or list of positions
                 
    Returns:
        tuple: A tuple containing a list of integers corresponding to the input string and the list of p_suc_inputs.
    """

    indexes = []
    if input == 'full':
        indexes = [int(i) for i in range(num_modes)]
    elif isinstance(input, list):
        indexes = np.array(input)
    else:
        n = int(input)
        n_gaps = n - 1
        total_spaces = num_modes - n
        av_spaces = (total_spaces)  // (n_gaps)
        left_over = (total_spaces) % (n_gaps)
        idx = 0
        positions = [0]
        for i in range(n_gaps):
            empties = av_spaces + (1 if i < left_over else 0)
            idx += empties + 1
            positions.append(idx)
        indexes = positions
    max_photons = len(indexes)    
    arr = np.zeros(num_modes, np.uint8)
    arr[indexes] = 1 
    arr = jnp.array(arr, dtype=jnp.uint8)
    probs = []
    if isinstance(p_suc_inputs, list):
        assert len(p_suc_inputs) == num_modes, "Length of p_suc_inputs must match num_modes"
        probs= jnp.array(p_suc_inputs)
    else:
        p = float(p_suc_inputs)
        probs = [p] * num_modes
    probs = jnp.array(probs, dtype=jnp.float32)
    return (arr, probs)
       







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











