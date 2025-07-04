# photonic_classifier/optimiser.py

import jax
import jax.numpy as jnp
from p_pack import globals
from p_pack import loss
from typing import List, Tuple, Any

# JAX itself just calculates gradients, but doesn't come with an optimizer.
# So I took the Adam optimizer from the old code.
# Also, to prevent the kernel from crashing, I had to use jax.lax.scan here instead of for-loops.
# And that requires writing a function for the loop iteration, and put most of the variables into a 
# list called 'carry'... Thanks, JAX.
@jax.jit
def adam_step(carry: tuple[jnp.array], step: int) -> Tuple[List[jnp.array], jnp.array]:
    """
    Performs a single Adam optimization step.

    This function is designed to be used with `jax.lax.scan` for efficient iteration.
    It calculates gradients for both phase and weight parameters and updates them
    according to the Adam algorithm.

    Args:
        carry (List[jnp.array]): A list containing the current state:
                             [params_phases, data_set, labels, params_weights,
                              m_phases, v_phases, m_weights, v_weights].
        step (int): The current optimization step number.

    Returns:
        Tuple[List[jnp.array], jnp.array]: A tuple containing the updated carry state and
                                   an array with the current step and loss value.
    """
    params_phases, data_set, labels, params_weights, m_phases, v_phases, m_weights, v_weights = carry
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    eta = globals.training_rate  # Learning rate

    loss_val = jax.lax.stop_gradient(loss.loss(params_phases, data_set, labels, params_weights))
    grad_params, grad_weights = jax.grad(loss.loss, argnums=(0, 3))(params_phases, data_set, labels, params_weights)

    # Update params
    m_phases = beta1 * m_phases + (1 - beta1) * jax.lax.stop_gradient(grad_params)
    v_phases = beta2 * v_phases + (1 - beta2) * jax.lax.stop_gradient(grad_params)**2
    m_hat_params = m_phases / (1 - beta1**step)
    v_hat_params = v_phases / (1 - beta2**step)
    params_phases = params_phases - eta * m_hat_params / (jnp.sqrt(v_hat_params) + eps)

    # Update weights
    m_weights = beta1 * m_weights + (1 - beta1) * jax.lax.stop_gradient(grad_weights)
    v_weights = beta2 * v_weights + (1 - beta2) * jax.lax.stop_gradient(grad_weights)**2
    m_hat_weights = m_weights / (1 - beta1**step)
    v_hat_weights = v_weights / (1 - beta2**step)
    params_weights = params_weights - eta * m_hat_weights / (jnp.sqrt(v_hat_weights) + eps)


    carry = (params_phases, data_set, labels, params_weights, m_phases, v_phases, m_weights, v_weights)
    return carry, jnp.array([step, loss_val])


#some function gpt spat out when rewriting will have a look later
def make_adam(step_size: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> Any:
    """
    A factory function that creates an Adam optimizer object.

    This optimizer is designed to work with JAX's tree-like data structures (pytrees),
    making it flexible for different parameter shapes.

    Args:
        step_size (float): The learning rate.
        beta1 (float): The exponential decay rate for the first moment estimates. Defaults to 0.9.
        beta2 (float): The exponential decay rate for the second moment estimates. Defaults to 0.999.
        eps (float): A small constant for numerical stability. Defaults to 1e-8.

    Returns:
        Any: An object with `.init(params)` and `.update(grads, state, params)` methods.
    """
    """
    Returns an object with .init(params) and .update(grads, state, params).
    """
    def init(params):
        m = jax.tree_map(jnp.zeros_like, params)
        v = jax.tree_map(jnp.zeros_like, params)
        t = jnp.array(0, dtype=jnp.int32)
        return m, v, t

    def update(grads, state, params):
        m, v, t = state
        t = t + 1
        m = jax.tree_multimap(lambda m_, g: beta1*m_ + (1-beta1)*g, m, grads)
        v = jax.tree_multimap(lambda v_, g: beta2*v_ + (1-beta2)*(g**2), v, grads)
        m_hat = jax.tree_map(lambda m_: m_/(1-beta1**t), m)
        v_hat = jax.tree_map(lambda v_: v_/(1-beta2**t), v)
        new_params = jax.tree_multimap(
            lambda p, mh, vh: p - step_size * mh/(jnp.sqrt(vh)+eps),
            params, m_hat, v_hat
        )
        return new_params, (m, v, t)

    return type("Adam", (), {"init": init, "update": update})