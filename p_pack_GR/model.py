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


def full_unitaries_superposition(
    phases: jnp.array,
    data_set: jnp.array,
    weights: jnp.array,
    mask,
    key,
    reupload_freq,
    shuffle_type=globals.shuffle_type,
) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array, jnp.ndarray, any]:
    """Construct the full unitary when the input photon is in superposition.

    The circuit layout and data re-uploading are identical to
    :func:`full_unitaries_data_reupload`, but instead of propagating
    explicit input photons through :func:`circ.measurement`, the function
    computes output probabilities by applying the final unitary to a
    state vector prepared as a uniform superposition over the modes
    specified by ``mask``.
    """
    depth = int(jax.lax.stop_gradient(phases).shape[0])

    if isinstance(reupload_freq, int):
        re_layers = list(range(0, depth, reupload_freq)) if reupload_freq > 0 else []
    else:
        re_layers = list(reupload_freq)
    reupload_set = set(re_layers)
    layer_order = {layer: idx for idx, layer in enumerate(re_layers)}

    if 0 in reupload_set:
        unitaries = circ.data_upload(weights[0, :] * data_set)
    else:
        single = circ.layer_unitary(phases, 0)
        unitaries = jnp.broadcast_to(single, (data_set.shape[0],) + single.shape)

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

            unitaries_data_reupload = circ.data_upload(weights[layer, :] * data_set_reupload)
            unitaries = unitaries_data_reupload @ unitaries
        else:
            unitaries = circ.layer_unitary(phases, layer) @ unitaries

    width = jax.lax.stop_gradient(unitaries).shape[-1]
    mask_arr = jnp.asarray(mask, dtype=jnp.float32)
    active = jnp.sum(mask_arr)
    # Prepare a single-photon state uniformly distributed over the
    # selected modes. ``mask`` indicates which modes participate in the
    # superposition.  At least one mode must be active.
    norm_state = (mask_arr / jnp.sqrt(active)).astype(jnp.complex64)
    output_states = unitaries @ norm_state
    mode_probs = jnp.abs(output_states) ** 2

    if globals.num_classes == 2:
        half = width // 2
        class_probs = jnp.stack([
            jnp.sum(mode_probs[:, :half], axis=1),
            jnp.sum(mode_probs[:, half:], axis=1),
        ], axis=1)
    else:
        class_list = [jnp.sum(mode_probs[:, c::globals.num_classes], axis=1) for c in range(globals.num_classes)]
        class_probs = jnp.stack(class_list, axis=1)

    n_p = jnp.array(1, dtype=jnp.int32)
    return unitaries, output_states, mode_probs, class_probs, n_p, key


@partial(jax.jit, static_argnames=['reupload_freq', 'shuffle_type'])
def predict_superposition(
    phases: jnp.array,
    data_set: jnp.array,
    weights: jnp.array,
    mask,
    key,
    reupload_freq,
    shuffle_type=globals.shuffle_type,
) -> Tuple[jnp.array, jnp.array, jnp.ndarray, any]:
    """Predict class probabilities for a superposed single-photon input state."""

    _, _, probs, class_probs, n_p, key = full_unitaries_superposition(
        phases, data_set, weights, mask, key, reupload_freq, shuffle_type
    )

    return probs, class_probs, n_p, key