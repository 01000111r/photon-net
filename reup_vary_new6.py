
from p_pack import globals as g


# ----- Global configuration -----
# training parameters
g.num_steps = 800
g.training_rate = 0.1
g.save_points = [50,100,200,400,800]  # steps at which to save model parameters

# reupload configuration
# g.reupload_freq = 4
g.reupload_freq = tuple([0,4,8]) # layers at which to re-upload data
g.reup_is_tuple = True

# How to shuffle data when re-uploading images.
# 0 - random permutation each upload (default)
# 1 - no shuffling, use the same ordering
# 2 - reverse the image every other upload layer
g.shuffle_type = 2

# dataset parameters
g.num_features = 5
# probability of success for each mode
g.p_suc_inputs = 1
# input positions configuration
g.input_positions = [0]
#parity type
g.use_symmetry_parity = False
# photon aim
g.aim = 3
# 0 to not discard, 1 to discard 
g.discard = 0
g.discard_condition = '!='
g.discard_range = None

# loss configuration
g.loss_function = 0
# initial phase value
g.phase_init_value = None


# Batch processing configuration
# 'full'  : use entire dataset each update
# 'mini'  : use mini-batches of size ``mini_batch_size``
# 'single': process one sample at a time
g.batch_mode = 'full'
g.mini_batch_size = 11000

# random seed
g.master_key = g.jax.random.PRNGKey(2)
g.phase_key = g.jax.random.PRNGKey(10)
g.shuffle_key = g.jax.random.PRNGKey(52)

# Key used when sampling new input photon positions each optimisation step.
g.position_key = g.jax.random.PRNGKey(7)
# If ``True`` a fresh set of input positions is sampled every update.
g.position_sampling: bool = False

g.dataset_name = "mnist_pca_test"
g.class_labels = [3, 5]
g.use_binary_labels = False
g.num_classes = len(g.class_labels)

#circuit dimensions
g.num_modes_circ = g.num_features * 2
g.depth = g.num_features * 2
# build input config
g.input_config = g.input_config_maker(g.input_positions, g.num_modes_circ, g.p_suc_inputs)

g.max_photons = 3 # maximum photon number for building probability calculating functions

from p_pack import pre_p, circ, model, loss, optimiser, train, utils





# ----- Data production function -----
from pathlib import Path

log_file = 'data_log'
folder_name = 'p1-reup-pos-vary-s2'
# outputs are written to the "work" directory under the user's home
folder = str(Path.home() / 'work' / folder_name)
# p_suc_list = [0, 1, 2, 3, 4, 5, 6 , 7, 8]
# varied_list= [0.1, -0.1, 0.01, -0.01]
# varied_list= [10, 10, 15, 20]
varied_list = [(0, 4, 9),
               (0, 5, 9),
               (1, 4, 8),
               (1, 5, 8),
               (4, 6, 8),
               (0, 2, 4),
               (3, 5, 7),
               (2, 4, 6),
               (1, 4, 8),
               (1, 5, 8)
               ]
# name of the global variable to modify during iteration
global_var_name = "reupload_freq"
# set to True if ``global_var_name`` should be treated as a PRNGKey seed
is_key = False
file_indent = 'f'
start_idx = 0


def data_prod_iterator(variable_list, globals_var_name, is_key, log_file, folder, file_indent, start_idx):
    """Iterate over variable_list, update global variable and run training."""
    for idx, var in enumerate(variable_list, start=start_idx):
        test_name = f"{idx}{file_indent}{var}.npz"
        global_name = f"{idx}{file_indent}{var}g.npz"

         
        if is_key:
            setattr(g, global_var_name, g.jax.random.PRNGKey(var))
        else:
            setattr(g, global_var_name, var)

        g.num_modes_circ = g.num_features * 2
        g.depth = g.num_features * 2
        g.input_config = g.input_config_maker(g.input_positions, g.num_modes_circ, g.p_suc_inputs)
        # ----- Load data -----
        train_set, train_labels, test_set, test_labels = g.final_load_data(g.num_features)

        # Initialize phases
        init_phases = circ.initialize_phases(g.depth, 2 * g.num_features, reupload_freq=g.reupload_freq)
        weights_data = g.jnp.ones(shape=[init_phases.shape[0], init_phases.shape[1]])

        if g.position_sampling:
            sub_pos = g.jax.random.fold_in(g.position_key, 0)
            mask = g.sample_input_config(sub_pos, g.input_config[0])
        else:
            mask = g.input_config[0]

        photon_loss_scale = float(1)
        initial_loss, (n0, key) = loss.loss(
            init_phases,
            train_set,
            train_labels,
            weights_data,
            photon_loss_scale,
            g.input_config,
            mask,
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
        print(var, "done")


if __name__ == "__main__":
    data_prod_iterator(varied_list, global_var_name, is_key, log_file, folder, file_indent, start_idx)