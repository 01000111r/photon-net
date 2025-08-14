
from p_pack import globals as g


# ----- Global configuration -----
# training parameters
g.num_steps = 500
g.training_rate = 0.1
g.save_points = [500]  # steps at which to save model parameters

# reupload configuration
g.reupload_freq = 4
# g.reupload_freq = tuple([0,4,8]) # layers at which to re-upload data
g.reup_is_tuple = False

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
g.batch_mode = 'mini'
g.mini_batch_size = 11000

# random seed
g.master_key = g.jax.random.PRNGKey(2)
g.phase_key = g.jax.random.PRNGKey(10)
g.shuffle_key = g.jax.random.PRNGKey(52)

# Key used when sampling new input photon positions each optimisation step.
g.position_key = g.jax.random.PRNGKey(7)
# If ``True`` a fresh set of input positions is sampled every update.
g.position_sampling: bool = False
# Optional mask restricting which input modes may be chosen when
# ``position_sampling`` is enabled.  Each entry corresponds to a circuit
# mode; ``1`` marks the mode as eligible while ``0`` forbids it.  By
# default all modes are allowed.
g.pos_allowed = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


g.use_input_superposition: bool = False

g.dataset_name = "mnist_pca_3"
g.class_labels = [3, 5, 7]
g.use_binary_labels = False
g.num_classes = len(g.class_labels)



g.max_photons = 3 # maximum photon number for building probability calculating functions

import importlib
import p_pack.pre_p as pre_p
import p_pack.circ as circ
import p_pack.model as model
import p_pack.loss as loss
import p_pack.optimiser as optimiser
import p_pack.train as train
import p_pack.utils as utils





# ----- Data production function -----
from pathlib import Path

log_file = 'data_log'
folder_name = 'p1-dim-vary-s2-c3'
# outputs are written to the "work" directory under the user's home
folder = str(Path.home() / 'work' / folder_name)
# p_suc_list = [0, 1, 2, 3, 4, 5, 6 , 7, 8]
# varied_list= [0.1, -0.1, 0.01, -0.01]
# varied_list= [10, 10, 15, 20]
varied_list = [3, 4, 5, 6, 7, 8, 9, 10]

# name of the global variable to modify during iteration
reupload_list = [2, 3, 4, 5, 6, 7, 8, 9]
file_indent = 'f'
start_idx = 0

# global_var_name = "num_features"
# # set to True if ``global_var_name`` should be treated as a PRNGKey seed
# is_key = False
# file_indent = 'p'
# start_idx = 0





def data_prod_iterator(feature_list, reupload_list, log_file, folder, file_indent, start_idx):
    """Iterate over paired lists of features and reupload freqs."""
    for idx, (feat, rfreq) in enumerate(zip(feature_list, reupload_list), start=start_idx):
        test_name = f"{idx}{file_indent}f{feat}r{rfreq}.npz"
        global_name = f"{idx}{file_indent}f{feat}r{rfreq}g.npz"

         
        g.num_features = feat
        g.reupload_freq = rfreq

        g.num_modes_circ = g.num_features * 2
        g.depth = g.num_features * 2
        g.input_config = g.input_config_maker(g.input_positions, g.num_modes_circ, g.p_suc_inputs)
        # ----- Load data -----
        train_set, train_labels, test_set, test_labels = g.final_load_data(g.num_features)

        # Rebuild circuit-dependent modules for the new mode count
        for mod in (circ, model, loss, optimiser, train, utils):
            importlib.reload(mod)

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
            g.shuffle_type,
            g.use_input_superposition
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
        print(feat, rfreq, "done")


if __name__ == "__main__":
    data_prod_iterator(varied_list, reupload_list, log_file, folder, file_indent, start_idx)