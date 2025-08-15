
from p_pack import globals as g


# ----- Global configuration -----
# training parameters
g.num_steps = 100
g.training_rate = 0.1

# How to shuffle data when re-uploading images.
# 0 - random permutation each upload (default)
# 1 - no shuffling, use the same ordering
# 2 - reverse the image every other upload layer
# 3 - always reverse the image
g.reupload_freq = 2
g.shuffle_type = 1

#circuit dimensions
g.num_modes_circ = 10
g.depth = 10

# dataset parameters
g.num_features = 5
# probability of success for each mode
g.p_suc_inputs = 1
# input positions configuration
g.input_positions = [4]
#parity type
g.use_symmetry_parity = False
# photon aim
g.aim = 1
# 0 to not discard, 1 to discard 
g.discard = 0
g.discard_condition = '!='
g.discard_range = None

# loss configuration
g.loss_function = 0
# initial phase value
g.phase_init_value = None

# batching
g.batch_mode = 'full'
g.mini_batch_size = 1000

# random seed
g.master_key = g.jax.random.PRNGKey(2)
g.phase_key = g.jax.random.PRNGKey(10)
g.shuffle_key = g.jax.random.PRNGKey(52)

# maximum photon number for discard logic
g.max_photons = 1

# build input config
g.input_config = g.input_config_maker(g.input_positions, g.num_modes_circ, g.p_suc_inputs)

from p_pack import pre_p, circ, model, loss, optimiser, train, utils

# ----- Load data -----
train_set, train_labels, test_set, test_labels = g.final_load_data(g.num_features)



# ----- Data production function -----
from pathlib import Path

log_file = 'data_log'
folder_name = 'symmetry-false-2p'
# outputs are written to the "work" directory under the user's home
folder = str(Path.home() / 'work' / folder_name)
#
# Example configurations to iterate over.  Each dictionary contains the
# globals that will be updated for that run.  Add additional dictionaries
# to ``run_configs`` as needed.
run_configs = [
    {"input_positions": [0,9], "shuffle_type": 0},
    {"input_positions": [0,9], "shuffle_type": 1},
    {"input_positions": [0,9], "shuffle_type": 2},
]

# Names of global variables that should be treated as PRNGKey seeds.  None of
# the variables above are seeds, so this is left empty but kept for
# compatibility with existing behaviour.
key_vars = set()

file_indent = 'f'
start_idx = 0


def data_prod_iterator(config_list, key_vars, log_file, folder, file_indent, start_idx):
    """Iterate over ``config_list`` updating globals for each run.

    Parameters
    ----------
    config_list : list of dict
        Each dictionary maps global variable names to the desired value for that
        run.
    key_vars : set
        Names of variables that should be initialised using ``jax.random.PRNGKey``.
    """
         
    for idx, cfg in enumerate(config_list, start=start_idx):
        var_str = "_".join(f"{k}-{str(v).replace(' ', '')}" for k, v in cfg.items())
        test_name = f"{idx}{file_indent}{var_str}.npz"
        global_name = f"{idx}{file_indent}{var_str}g.npz"

        for name, value in cfg.items():
            if name in key_vars:
                setattr(g, name, g.jax.random.PRNGKey(value))
            else:
                setattr(g, name, value)

        g.input_config = g.input_config_maker(g.input_positions, g.num_modes_circ, g.p_suc_inputs)

        # Initialize phases
        init_phases = circ.initialize_phases(g.depth, 2 * g.num_features)
        weights_data = g.jnp.ones(shape=[init_phases.shape[0], init_phases.shape[1]])

        photon_loss_scale = float(1)
        initial_loss, (n0, key) = loss.loss(
            init_phases,
            train_set,
            train_labels,
            weights_data,
            photon_loss_scale,
            g.input_config,
            g.master_key,
            g.loss_function,
            g.aim,
            g.reupload_freq,
            g.shuffle_type
        )
        init_carry = (
            init_phases,
            train_set,
            train_labels,
            weights_data,
            photon_loss_scale,
            0.0 * init_phases,
            0.0 * init_phases,
            0.0 * weights_data,
            0.0 * weights_data,
            0.0 * photon_loss_scale,
            0.0 * photon_loss_scale,
            key,
            initial_loss,
        )

        utils.save_run(log_file, folder, test_name, global_name, init_carry)
        print(cfg, "done")


if __name__ == "__main__":
    data_prod_iterator(run_configs, key_vars, log_file, folder, file_indent, start_idx)