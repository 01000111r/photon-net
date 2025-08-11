# photonic_classifier/model.py

import jax
import jax.numpy as jnp
from p_pack import circ, globals
from typing import Tuple, List
from functools import partial

#key = jax.random.PRNGKey(12)

def full_unitaries_data_reupload(phases: jnp.array, data_set: jnp.array, weights: jnp.array, input_config, mask, key, reupload_freq, shuffle_type=globals.shuffle_type) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
    """
    Constructs the full unitary transformation of the circuit with data re-uploading.

    This function simulates the entire photonic circuit, layer by layer. It alternates
    between applying trainable unitary layers and data-re-uploading layers based on
    the `reupload_freq` global setting.

    Args:
        phases (jnp.array): The trainable phase parameters of the circuit.
        data_set (jnp.array): The input data samples.
        weights (jnp.array): The weights applied to the data during re-uploading steps.

    Returns:
        Tuple[jnp.array, jnp.array, jnp.array, jnp.array]: A tuple containing:
            - The final full unitary matrices for each sample.
            - The sub-unitaries relevant to the input photon states.
            - The raw probabilities for all output label combinations.
            - The aggregated binary probability for the +1 outcome.
    """
    # Depth of the trainable part.
    depth = int(jax.lax.stop_gradient(phases).shape[0])

    # Determine which layers perform data re-uploading.
    if isinstance(reupload_freq, int):
        re_layers = list(range(0, depth, reupload_freq)) if reupload_freq > 0 else []

    else:
        re_layers = list(reupload_freq)
    reupload_set = set(re_layers)
    layer_order = {layer: idx for idx, layer in enumerate(re_layers)}

    # First layer: either data upload or trainable unitary.
    if 0 in reupload_set:
        unitaries = circ.data_upload(weights[0, :] * data_set)
    else:
        single = circ.layer_unitary(phases, 0)
        unitaries = jnp.broadcast_to(single, (data_set.shape[0],) + single.shape)

    #print('First layer shape', first_layers)
    #print('First layers shape', first_layers)
    for layer in range(1, depth):
        if layer in reupload_set:    
            if shuffle_type == 0:
                key2 = jax.random.fold_in(globals.shuffle_key, layer)
                temp = jax.random.permutation(key2, data_set.shape[1])
                temp = jax.lax.stop_gradient(temp)
                data_set_reupload = data_set[:, temp]
            elif shuffle_type == 1:
                data_set_reupload = data_set
            elif shuffle_type == 2:
                if (layer_order[layer] % 2) == 1:
                    data_set_reupload = data_set[:, ::-1]
                else:
                    data_set_reupload = data_set
            elif shuffle_type == 3:
                data_set_reupload = data_set[:, ::-1]
            else:
                data_set_reupload = data_set
            
            #temp_permutation = data_set_reupload[:10, :3]
            #print(temp_permutation)


            unitaries_data_reupload = circ.data_upload(weights[layer,:]* data_set_reupload)
            #print('Reupload layer', layer, 'shape', unitaries_data_reupload)
            unitaries = unitaries_data_reupload @ unitaries
        else:
            unitaries = circ.layer_unitary(phases, layer) @ unitaries
 
    #now we have full unitaries for all the differnt circuits that corresponf to each image upload, all same parameters though, each reupload layer have different weights but the weights are the same for all images.

    # Extract the probabilities of the output states.
    sub_unitaries, label_probs, class_probs, n_p, key = circ.measurement(unitaries, mask, input_config, key)
    #print(label_probs[:10, :])
    #print(binary_probs_plus[:10,:])
   
    return unitaries, sub_unitaries, label_probs, class_probs, n_p, key


@partial(jax.jit, static_argnames=['reupload_freq', 'input_config', 'shuffle_type'])
def predict_reupload(phases: jnp.array, data_set: jnp.array, weights: jnp.array, input_config, mask, key, reupload_freq, shuffle_type=globals.shuffle_type) -> Tuple[jnp.array, jnp.array]:
    """
    Performs prediction using the photonic circuit model.

    This function computes the full circuit transformation and then adjusts the final
    binary probability. The output is mapped such that probabilities > 0.5 are
    positive and probabilities < 0.5 are negative, representing the two classes.

    Args:
        phases (jnp.array): The trainable phase parameters of the circuit.
        data_set (jnp.array): The input data samples.
        weights (jnp.array): The weights applied to the data during re-uploading steps.

    Returns:
        Tuple[jnp.array, jnp.array]: A tuple containing:
            - probs: The raw probabilities for all output combinations.
            - adjusted_binary_probs: The final binary prediction, adjusted to be positive or negative.
    """

    _, _, probs, class_probs, n_p, key = full_unitaries_data_reupload(phases, data_set, weights, input_config, mask, key, reupload_freq, shuffle_type)

    return probs, class_probs, n_p, key
