from p_pack import globals as g


# ----- Global configuration -----
# training parameters
g.num_steps = 800
g.training_rate = 0.1
g.save_points = [100, 500, 800]  # steps at which to save model parameters

# reupload configuration
g.reupload_freq = 4  # data upload only at layer 0

# g.reupload_freq = tuple([0,4,8]) # layers at which to re-upload data
g.reup_is_tuple = False

# How to shuffle data when re-uploading images.
# 0 - random permutation each upload (default)
# 1 - no shuffling, use the same ordering
# 2 - reverse the image every other upload layer
g.shuffle_type = 1


#controlling extra features

#toggling the option to upload more features
g.use_extra_features = True

#total number of features (should equal num_features if use_extra_features = False) 
g.total_features = 10

#this num_features controls the number of modes on constructed circuit
g.num_features = 5

#this reuploads the extra features on the layer after the primary features (has not been tested with p_and_q_encoding=True)
g.reupload_secondary = False
g.p_and_q_encoding= True



# probability of success for each mode
g.p_suc_inputs = 1
# input positions configuration
g.input_positions = [4]
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

# 0 = MSE, 1 = Cross-Entropy
g.loss_metric = 0

#coherence slider, 1.0 = full interfernece 0.0= none (I think the logic should be right but maybe check)
g.coherence = 0.0


# initial phase value
g.phase_init_value = 0.0

# this freezes the circuit
g.freeze_phases = False

# Controls which readout method is used after the photonic circuit.
# 0 - parity readout (fixed, no training)
# 1 - trained linear readout (softmax classifier)
g.readout_type=1



# Batch processing configuration
# 'full'  : use entire dataset each update
# 'mini'  : use mini-batches of size ``mini_batch_size``
# 'single': process one saMple at a time
g.batch_mode = 'mini'
g.mini_batch_size = 512

# random seed
g.master_key = g.jax.random.PRNGKey(6)
g.phase_key = g.jax.random.PRNGKey(13)
g.shuffle_key = g.jax.random.PRNGKey(51)

# Key used when sampling new input photon positions each optimisation step.
g.position_key = g.jax.random.PRNGKey(7)
# If ``True`` a fresh set of input positions is sampled every update.
g.position_sampling: bool = False
# Optional mask restricting which input modes may be chosen when
# ``position_sampling`` is enabled.  Each entry corresponds to a circuit
# mode; ``1`` marks the mode as eligible while ``0`` forbids it.  By
# default all modes are allowed.
g.pos_allowed = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

g.dataset_name = "mnist_pca_10"
g.class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
g.use_binary_labels = False
g.num_classes = len(g.class_labels)

g.use_input_superposition: bool = False

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
folder_name = 'mnist_dist_pandq'
# outputs are written to the "work" directory under the user's home
folder = str(Path.home() / 'work' / folder_name)
# p_suc_list = [0, 1, 2, 3, 4, 5, 6 , 7, 8]
# varied_list= [0.1, -0.1, 0.01, -0.01]
# varied_list= [10, 10, 15, 20]
varied_list = [[4], [4,5], [4,5,6]]

# # name of the global variable to modify during iteration
# reupload_list = [7,8]
# file_indent = 'p'
# start_idx = 0

global_var_name = "input_positions"
# set to True if ``global_var_name`` should be treated as a PRNGKey seed
is_key = False
file_indent = 'p'
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

        if g.use_extra_features:
            train_total, train_labels, test_total, test_labels = g.final_load_data(g.total_features)
            train_set = train_total   # pass all features together, model.py slices internally
            test_set  = test_total
           

            allocation = g.compute_extra_layer_allocation(g.total_features, g.num_features, p_and_q=g.p_and_q_encoding)

            if g.reupload_secondary:
                extra_layer_cols = {layer: (start, end) for layer, (start, end) in allocation.items()}
                offsets = [(layer, start, end) for layer, (start, end) in allocation.items()]
                if isinstance(g.reupload_freq, int):
                    re_layers = list(range(0, g.depth, g.reupload_freq))[1:]
                else:
                    re_layers = list(g.reupload_freq)[1:]
                for re_layer in re_layers:
                    for initial_layer, start, end in offsets:
                        new_layer = re_layer + initial_layer
                        if new_layer < g.depth:
                            extra_layer_cols[new_layer] = (start, end)
                g.extra_upload_layers = list(extra_layer_cols.keys())
                g.extra_layer_cols = extra_layer_cols
            else:
                g.extra_upload_layers = list(allocation.keys())
                g.extra_layer_cols = {layer: (start, end) for layer, (start, end) in allocation.items()}



        else:
            train_set, train_labels, test_set, test_labels = g.final_load_data(g.num_features)
            g.extra_upload_layers = []
            g.extra_layer_cols = {}
            g.extra_data = None

        load_features = g.total_features if g.use_extra_features else g.num_features
        train_file = g.get_mnist_csv_filepath("train", load_features)
        test_file  = g.get_mnist_csv_filepath("test", load_features)


        # Rebuild modules
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

        #Initalising tactically
        readout_weights = 0.01 * g.jax.random.normal(g.jax.random.PRNGKey(123), (circ.max_n_combos, g.num_classes))

        #Intialising with 0s
        #readout_weights = g.jnp.zeros((circ.max_n_combos, g.num_classes))

        
        
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
            g.use_input_superposition,
            readout_weights,
            g.readout_type
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
            readout_weights, 0.0*readout_weights, 0.0*readout_weights,
        )


        carry, _, _, _ = utils.save_run(log_file, folder, test_name, global_name, init_carry)

if __name__ == "__main__":
    data_prod_iterator(varied_list, global_var_name, is_key, log_file, folder, file_indent, start_idx)