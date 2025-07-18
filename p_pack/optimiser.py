# photonic_classifier/optimiser.py

from functools import partial
import jax
import jax.numpy as jnp
from p_pack import globals
from p_pack import loss
from typing import List, Tuple, Any

@partial(jax.jit, static_argnames=("batch_mode", "mini_batch_size"))
def _select_batch(ds: jnp.ndarray, lb: jnp.ndarray, key: jax.random.PRNGKey, batch_mode, mini_batch_size) -> tuple[jnp.ndarray, jnp.ndarray, jax.random.PRNGKey]:
    """Return a batch of data according to .batch_mode."""
    mode = batch_mode
    if mode == "full":
        return ds, lb, key
    if mode == "mini":
        size = mini_batch_size
    elif mode == "single":
        size = 1
    else:
        size = ds.shape[0]

    key, sub = jax.random.split(key)
    idx = jax.random.choice(sub, ds.shape[0], shape=(size,), replace=False)
    return ds[idx], lb[idx], key



@partial(jax.jit, static_argnames=['discard', 'aim', 'cmp', 'loss_function', 'range_vals', 'batch_mode', 'mini_batch_size'])
def adam_step(
    carry,
    step,
    discard,
    aim,
    cmp,
    input_config,
    loss_function,
    training_rate,
    range_vals, 
    batch_mode,
    mini_batch_size
):
    """
    ccarry = (
        pp, ds, lb, pw, pa,
        mp, vp, mw, vw, ma, va,
        key, last_loss
    )
    step  = scalar int
    pp    = params_phases
    ds    = data_set
    lb    = labels
    pw    = params_weights
    pa    = params_alpha
    mp    = m_phases
    vp    = v_phases
    mw    = m_weights
    vw    = v_weights
    ma    = m_alpha
    mv    = v_alpha
    key  = PRNGKey
    last_loss = last loss value

    returns (new_carry, (out, did_update))
      where out       = [step, loss_val]
            did_update = 1 if we updated, 0 if we skipped
    """
    pp, ds, lb, pw, pa, mp, vp, mw, vw, ma, va, key, last_loss = carry

    # 0) Select a batch of data
    ds_b, lb_b, key = _select_batch(ds, lb, key, batch_mode, mini_batch_size)

    # 1) Evaluate loss & grads, get back a fresh PRNGKey
    (loss_val, (n_p, new_key)), (g_pp, g_pw, g_pa) = jax.value_and_grad(
        loss.loss,
        argnums=(0, 3, 4),
        has_aux=True,
    )(pp, ds_b, lb_b, pw, pa, input_config, key, loss_function, aim)

    # 2) Decide whether to skip:
    #    only skip if discard==1 and the photon condition is met
    cmp_funcs = {
        '!=': jnp.not_equal,
        '==': jnp.equal,
        '<': jnp.less,
        '<=': jnp.less_equal,
        '>': jnp.greater,
        '>=': jnp.greater_equal,}
    
    if cmp == 'range' and range_vals is not None:
        lower, upper = range_vals
        condition = jnp.logical_and(
            jnp.greater_equal(n_p, jnp.array(lower, dtype=n_p.dtype)),
            jnp.less_equal(n_p, jnp.array(upper, dtype=n_p.dtype))
        )
    else:
        cmp_fn = cmp_funcs.get(cmp, jnp.not_equal)
        condition = cmp_fn(n_p, jnp.array(aim, dtype=n_p.dtype))
        
    dynamic_bool = jnp.logical_and(
        jnp.array(discard == 1),
        condition
    )

    no_photons = jnp.all(n_p == 0)

    skip_step_bool = jnp.logical_or(dynamic_bool, no_photons)
    
    # 3a) Skip branch: replay the old carry but swap in the fresh key
    def skip_fn(carry):
        pp, ds, lb, pw, pa, mp, vp, mw, vw, ma, va, _, last = carry
        new_carry = (
            pp, ds, lb, pw, pa,
            mp, vp, mw, vw, ma, va,
            new_key, last
        )
        out       = jnp.array([step, last], dtype=jnp.float32)
        return new_carry, (out, jnp.array(0, dtype=jnp.int32), n_p)

    # 3b) Update branch: do the Adam update, record new loss & key
    def update_fn(carry):
        pp, ds, lb, pw, pa, mp, vp, mw, vw, ma, va, _, _ = carry
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        eta = training_rate

        # phases
        mp_new  = beta1 * mp  + (1 - beta1) * g_pp
        vp_new  = beta2 * vp  + (1 - beta2) * (g_pp ** 2)
        mp_hat  = mp_new / (1 - beta1**step)
        vp_hat  = vp_new / (1 - beta2**step)
        pp_new  = pp - eta * mp_hat / (jnp.sqrt(vp_hat) + eps)

        # weights
        mw_new  = beta1 * mw  + (1 - beta1) * g_pw
        vw_new  = beta2 * vw  + (1 - beta2) * (g_pw ** 2)
        mw_hat  = mw_new / (1 - beta1**step)
        vw_hat  = vw_new / (1 - beta2**step)
        pw_new  = pw - eta * mw_hat / (jnp.sqrt(vw_hat) + eps)

        # alpha parameter
        ma_new = beta1 * ma + (1 - beta1) * g_pa
        va_new = beta2 * va + (1 - beta2) * (g_pa ** 2)
        ma_hat = ma_new / (1 - beta1**step)
        va_hat = va_new / (1 - beta2**step)
        pa_new = pa - eta * ma_hat / (jnp.sqrt(va_hat) + eps)

        new_carry = (
            pp_new, ds, lb, pw_new, pa_new,
            mp_new, vp_new, mw_new, vw_new, ma_new, va_new,
            new_key, loss_val
        )
        out       = jnp.array([step, loss_val], dtype=jnp.float32)
        return new_carry, (out, jnp.array(1, dtype=jnp.int32), n_p)

    # 4) Branch
    (new_carry, (out, did_update, p_num)) = jax.lax.cond(skip_step_bool, skip_fn, update_fn, carry)
    return new_carry, (out, did_update, p_num)








# # JAX itself just calculates gradients, but doesn't come with an optimizer.
# # So I took the Adam optimizer from the old code.
# # Also, to prevent the kernel from crashing, I had to use jax.lax.scan here instead of for-loops.
# # And that requires writing a function for the loop iteration, and put most of the variables into a 
# # list called 'carry'... Thanks, JAX.
# @jax.jit
# def adam_step(carry: tuple[jnp.array], step: int) -> Tuple[List[jnp.array], jnp.array]:
#     """
#     Performs a single Adam optimization step.

#     This function is designed to be used with `jax.lax.scan` for efficient iteration.
#     It calculates gradients for both phase and weight parameters and updates them
#     according to the Adam algorithm.

#     Args:
#         carry (List[jnp.array]): A list containing the current state:
#                              [params_phases, data_set, labels, params_weights,
#                               m_phases, v_phases, m_weights, v_weights].
#         step (int): The current optimization step number.

#     Returns:
#         Tuple[List[jnp.array], jnp.array]: A tuple containing the updated carry state and
#                                    an array with the current step and loss value.
#     """
#     params_phases, data_set, labels, params_weights, m_phases, v_phases, m_weights, v_weights, key = carry


#     beta1 = 0.9
#     beta2 = 0.999
#     eps = 1e-8

#     eta = globals.training_rate  # Learning rate

#     # loss_val = jax.lax.stop_gradient(loss.loss(params_phases, data_set, labels, params_weights, globals.input_config, key)[0])
#     # n_p = jax.lax.stop_gradient(loss.loss(params_phases, data_set, labels, params_weights, globals.input_config, key)[1])

#     # grad_params, grad_weights = jax.grad(loss.loss, argnums=(0, 3))(params_phases, data_set, labels, params_weights)

#     (loss_val, (n_p, key)), (grad_params, grad_weights) = \
#         jax.value_and_grad(
#             loss.loss,
#             argnums=(0, 3),
#             has_aux=True
#         )(params_phases,
#           data_set,
#           labels,
#           params_weights,
#           globals.input_config,
#           key)


#     if n_p != globals.aim and globals.discard == 1:
#         return carry, jnp.array([step, loss_val])
    
#     #grad_params, grad_weights = jax.grad(lambda pp, ds, lb, pw, ic: loss.loss(pp, ds, lb, pw, ic)[0],argnums=(0, 3))(params_phases, data_set, labels, params_weights, input_config)
    
#     # Update params
#     m_phases = beta1 * m_phases + (1 - beta1) * jax.lax.stop_gradient(grad_params)
#     v_phases = beta2 * v_phases + (1 - beta2) * jax.lax.stop_gradient(grad_params)**2
#     m_hat_params = m_phases / (1 - beta1**step)
#     v_hat_params = v_phases / (1 - beta2**step)
#     params_phases = params_phases - eta * m_hat_params / (jnp.sqrt(v_hat_params) + eps)

#     # Update weights
#     m_weights = beta1 * m_weights + (1 - beta1) * jax.lax.stop_gradient(grad_weights)
#     v_weights = beta2 * v_weights + (1 - beta2) * jax.lax.stop_gradient(grad_weights)**2
#     m_hat_weights = m_weights / (1 - beta1**step)
#     v_hat_weights = v_weights / (1 - beta2**step)
#     params_weights = params_weights - eta * m_hat_weights / (jnp.sqrt(v_hat_weights) + eps)


#     carry = (params_phases, data_set, labels, params_weights, m_phases, v_phases, m_weights, v_weights, key)
#     return carry, jnp.array([step, loss_val])


#some function gpt spat out when rewriting will have a look later
# def make_adam(step_size: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> Any:
#     """
#     A factory function that creates an Adam optimizer object.

#     This optimizer is designed to work with JAX's tree-like data structures (pytrees),
#     making it flexible for different parameter shapes.

#     Args:
#         step_size (float): The learning rate.
#         beta1 (float): The exponential decay rate for the first moment estimates. Defaults to 0.9.
#         beta2 (float): The exponential decay rate for the second moment estimates. Defaults to 0.999.
#         eps (float): A small constant for numerical stability. Defaults to 1e-8.

#     Returns:
#         Any: An object with `.init(params)` and `.update(grads, state, params)` methods.
#     """
#     """
#     Returns an object with .init(params) and .update(grads, state, params).
#     """
#     def init(params):
#         m = jax.tree_map(jnp.zeros_like, params)
#         v = jax.tree_map(jnp.zeros_like, params)
#         t = jnp.array(0, dtype=jnp.int32)
#         return m, v, t

#     def update(grads, state, params):
#         m, v, t = state
#         t = t + 1
#         m = jax.tree_multimap(lambda m_, g: beta1*m_ + (1-beta1)*g, m, grads)
#         v = jax.tree_multimap(lambda v_, g: beta2*v_ + (1-beta2)*(g**2), v, grads)
#         m_hat = jax.tree_map(lambda m_: m_/(1-beta1**t), m)
#         v_hat = jax.tree_map(lambda v_: v_/(1-beta2**t), v)
#         new_params = jax.tree_multimap(
#             lambda p, mh, vh: p - step_size * mh/(jnp.sqrt(vh)+eps),
#             params, m_hat, v_hat
#         )
#         return new_params, (m, v, t)

#     return type("Adam", (), {"init": init, "update": update})