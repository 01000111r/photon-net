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

# Traditionally this was an integer specifying how often data should be
# re-inserted into the circuit, i.e. a new data layer every
# ``reupload_freq`` layers.  To allow for more flexible layouts this value
# can now also be a sequence of explicit layer indices.  When a sequence is
# provided, data uploading layers are placed exactly at those positions.  If
# an integer is given the old behaviour is retained.
reupload_freq = 3
#reupload_freq = tuple([0,4,8])
reup_is_tuple = False

# How to shuffle data when re-uploading images.
# 0 - random permutation each upload (default)
# 1 - no shuffling, use the same ordering
# 2 - reverse the image every other upload layer
shuffle_type: int = 0

# The number of modes in the photonic circuit.
num_modes_circ: int = 10

# The number of layers in the photonic circuit.
depth: int = 10

# The user only needs to modify this FEATURE_SIZE.
num_features = 3

# probability of sucess for each mode /source looss
p_suc_inputs = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]

input_positions = '3'

# If ``True`` parity checks in :mod:`p_pack.circ` are performed
# relative to the middle of the circuit.  When ``False`` the
# parity is evaluated starting from index 0 as before.
use_symmetry_parity = False

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

master_key = jax.random.PRNGKey(0)
phase_key = jax.random.PRNGKey(20)
# Global PRNG key controlling shuffling when ``shuffle_type == 0``
shuffle_key = jax.random.PRNGKey(42)

# Key used when sampling new input photon positions each optimisation step.
position_key = jax.random.PRNGKey(7)
# If ``True`` a fresh set of input positions is sampled every update.
position_sampling: bool = False
# If set to a float value, all phases are initialised to this
# constant instead of random values.  ``None`` keeps the random
# initialisation behaviour.
phase_init_value: float = None

# A list of training steps at which the current model parameters
# should be checkpointed.  Each entry is interpreted as a 1-indexed
# step number during optimisation.  By default no intermediate
# checkpoints are saved.
save_points: list[int] = []



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
    arr = [0] * num_modes
    for idx in indexes:
        arr[int(idx)] = 1
    probs = []
    if isinstance(p_suc_inputs, list):
        assert (len(p_suc_inputs) == num_modes), "Length of p_suc_inputs must match num_modes"
        probs = [float(p) for p in p_suc_inputs]
    else:
        p = float(p_suc_inputs)
        probs = [p] * num_modes
    return (tuple(arr), tuple(probs))
       







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

def rescale_data(data_set: np.ndarray, min_val: float = -(np.pi)/2, max_val: float = (np.pi)/2) -> np.ndarray:
    """
    Rescales a dataset to a specified range [min_val, max_val].

    The function first normalizes the data to the [0, 1] range based on its
    min and max values, and then scales it to the target range.

    Args:
        data_set (np.ndarray): The input data to be rescaled.
        min_val (float): The minimum value of the target range. Defaults to -pi/2.
        max_val (float): The maximum value of the target range. Defaults to pi/2.

    Returns:
        np.ndarray: The rescaled data.
    """
    min_data = np.copysign(np.ceil(np.abs(jnp.min(data_set))), jnp.min(data_set))
    max_data = np.copysign(np.ceil(np.abs(jnp.max(data_set))), jnp.max(data_set))
    
    # Rescale the data to the range [-pi/2, pi/2]
    rescaled_data = (data_set - min_data) / (max_data - min_data)
    # Now scale it to the desired range
    rescaled_data = rescaled_data * (max_val - min_val) + min_val
    return rescaled_data

def final_load_data(num_feature):
    split_data = load_and_split_data(num_feature)
    train_set, train_labels, test_set, test_labels = split_data
    train_set = rescale_data(train_set, min_val = -(np.pi)/2, max_val = (np.pi/2))
    test_set = rescale_data(test_set, min_val = -(np.pi)/2, max_val = (np.pi/2))
    return train_set, train_labels, test_set, test_labels


@jax.jit
def sample_input_config(key: jax.random.PRNGKey, mask: jnp.ndarray) -> jnp.ndarray:
    """Shuffle a base presence ``mask`` to generate random input positions.

    Args:
        key: PRNG key used for shuffling.
        mask: Base binary mask indicating photon locations.

    Returns:
        jnp.ndarray: A new presence mask with the ones randomly permuted.
    """

    mask = jnp.asarray(mask, dtype=jnp.int32)
    perm = jax.random.permutation(key, mask.shape[0])
    return tuple(mask[perm])











