import jax
import jax.numpy as jnp
from p_pack import model
from p_pack import globals
import jax
from functools import partial


@partial(jax.jit, static_argnames=['loss_function', 'reupload_freq', 'input_config', 'shuffle_type' ])
def loss(phases: jnp.array,
         data_set: jnp.array,
         labels: jnp.array,
         weights: jnp.array,
         photon_loss_scale: float,
         input_config,
         key,
         loss_function,
         aim,
         reupload_freq,
         shuffle_type
        ) -> jnp.array:
    """
    Calculates the mean squared error loss for the photonic classifier.

    The loss is computed based on the squared difference between the ideal output (1.0)
    and the model's adjusted predictions. The predictions are adjusted based on the
    true labels: for a label of +1, the raw prediction is used, and for a label of -1,
    1.0 minus the raw prediction is used. 

    Args:
        phases (jnp.array): The trainable phase parameters of the circuit.
        data_set (jnp.array): The input data samples.
        labels (jnp.array): The true labels for the data samples, expected to be +1 or -1.
        weights (jnp.array): The weights applied to the data during re-uploading steps.

    Returns:
        jnp.array: The mean squared error loss as a scalar JAX array.
    """
    num_samples = jax.lax.stop_gradient(data_set).shape[0]
   
    _, binary_predictions_plus, n_p, key = model.predict_reupload(phases, data_set, weights, input_config, key, reupload_freq, shuffle_type)


    binary_predictions_plus = binary_predictions_plus.squeeze() # to match shapes
    binary_predictions_plus = jnp.abs(binary_predictions_plus) # as mentioned previously, we dont use the negative probs
    # Adjust predictions based on labels: if label == +1, keep it; else flip it
    adjusted_predictions = jnp.where(labels == 1, binary_predictions_plus, (1.0 - (binary_predictions_plus)))


    
    if loss_function == 0:
        # No scaling
        weight = 1.0
    elif loss_function == 1:
        # Scaling by photon loss scale - not sure if this is the best way to do it, maybe should be in optimisation step?
        weight = jnp.exp(photon_loss_scale * ((jnp.array(n_p, dtype=jnp.float32) / jnp.array(aim, dtype=jnp.float32)) - 1))

    loss = (weight * (1.0 - adjusted_predictions)**2).mean()

    return loss, (n_p, key)
