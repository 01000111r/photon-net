# photonic_classifier/model.py

import jax
import jax.numpy as jnp
from p_pack import circ, globals
from typing import Tuple, List

#key = jax.random.PRNGKey(12)
def full_unitaries_data_reupload(phases: jnp.array, data_set: jnp.array, weights: jnp.array) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
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
    depth = jax.lax.stop_gradient(phases).shape[0]

    # Please note that we broadcast over images in the data set. 
    # The convention is that only the last two indices are used for matrix operations, 
    # the others are broadcasting dimensions used for batches of images.

    first_layers = circ.data_upload(weights[0,:]*data_set)
    unitaries = first_layers
    #print('First layer shape', first_layers)
    #print('First layers shape', first_layers)
    for layer in range(1,depth): 
        
        if (layer)%globals.reupload_freq != 0: # every 'reupload_freq' layer is a upload layer 
            unitaries = circ.layer_unitary(phases, layer) @ unitaries
            #print('Layer', layer, 'shape', unitaries)
        # 'layer' is the layer index in the trainable part, starting from 0.
        else:        

            key = jax.random.PRNGKey(layer) 
            temp = jax.random.permutation(key, data_set.shape[1])
            temp = jax.lax.stop_gradient(temp)
            #temp = jnp.arange(data_set.shape[0])
            data_set_reupload = data_set[:,temp]
            
            #temp_permutation = data_set_reupload[:10, :3]
            #print(temp_permutation)


            unitaries_data_reupload = circ.data_upload(weights[layer,:]* data_set_reupload)
            #print('Reupload layer', layer, 'shape', unitaries_data_reupload)
            unitaries = unitaries_data_reupload @ unitaries

 

    # Extract the probabilities of the output states.
    sub_unitaries, _, label_probs, binary_probs_plus = circ.measurement(unitaries, num_photons = 3)
    #print(label_probs[:10, :])
    #print(binary_probs_plus[:10,:])
   
    return unitaries, sub_unitaries, label_probs, binary_probs_plus


@jax.jit
def predict_reupload(phases: jnp.array, data_set: jnp.array, weights: jnp.array) -> Tuple[jnp.array, jnp.array]:
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

    _, _, probs, binary_probs_plus = full_unitaries_data_reupload(phases, data_set, weights)
    adjusted_binary_probs = jnp.where(binary_probs_plus > 0.5, binary_probs_plus,  - binary_probs_plus)
    
    return probs, adjusted_binary_probs
