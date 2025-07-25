
from p_pack import globals as g


# ----- Global configuration -----
# training parameters
g.num_steps = 10
g.training_rate = 0.1

# circuit parameters
g.reupload_freq = 2
g.num_modes_circ = 10
g.depth = 10

# dataset parameters
g.num_features = 5
# probability of success for each mode
g.p_suc_inputs = 1
# input positions configuration
g.input_positions = [0]
# photon aim
g.aim = 1
# discard settings
g.discard = 0
g.discard_condition = '!='
g.discard_range = None

# loss configuration
g.loss_function = 0
# initial phase value
g.phase_init_value = 0.1

# batching
g.batch_mode = 'full'
g.mini_batch_size = 1000

# random seed
g.master_key = g.jax.random.PRNGKey(2)

# maximum photon number for discard logic
g.max_photons = 3

# build input config
g.input_config = g.input_config_maker(g.input_positions, g.num_modes_circ, g.p_suc_inputs)

from p_pack import pre_p, circ, model, loss, optimiser, train, utils

# ----- Load data -----
train_set, train_labels, test_set, test_labels = g.final_load_data(g.num_features)



# ----- Data production function -----
from pathlib import Path

log_file = 'data_log'
# outputs are written to the "work" directory under the user's home
folder = str(Path.home() / 'work' / 'test-reup-fix')
# p_suc_list = [0, 1, 2, 3, 4, 5, 6 , 7, 8]
p_suc_list = [0, 1, 2]
# variable to modify during iteration
global_var = g.reupload_freq
file_indent = 'f'


def data_prod_iterator(variable_list, globals_v, log_file, folder, file_indent):
    """Iterate over variable_list, update global variable and run training."""
    for var in variable_list:
        test_name = f"{file_indent}{var}.npz"
        global_name = f"{file_indent}{var}g.npz"

        # set the global variable for this run
        globals_v = var
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
    data_prod_iterator(p_suc_list, global_var, log_file, folder, file_indent)