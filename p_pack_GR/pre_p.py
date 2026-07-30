import os
import pandas as pd
import jax.numpy as jnp
import numpy as np



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

# Initializing phases over the full interval [0,2pi] invites barren plateaus. It is much  
# better to initialize close to 0.

def load_mnist_35(root_dir: str, feature_dim: int, split: str = "train") -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Loads and preprocesses a subset of the MNIST dataset containing only digits 3 and 5.

    This function reads a specific CSV file, extracts features and labels,
    and then rescales the feature data to the range [-pi/2, pi/2].
    Initializing phases over the full interval [0, 2pi] can lead to barren plateaus,
    so initializing close to 0 is preferred.

    Args:
        root_dir (str): The directory where the dataset CSV is located.
        feature_dim (int): The number of features (pixels) in the dataset.
        split (str): The dataset split to load, e.g., "train" or "test". Defaults to "train".

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing the scaled feature data (X_scaled)
                                         and the labels (y).
    """
    fname = f"mnist_3-5_{feature_dim}d_{split}.csv"
    path = os.path.join(root_dir, fname)
    df = pd.read_csv(path)
    arr = jnp.array(df.values)
    X = arr[:, :feature_dim]
    y = arr[:, feature_dim].astype(jnp.int32)
    X_scaled = rescale_data(X, min_val=-(np.pi)/2, max_val=(np.pi)/2)
    return X_scaled, y
