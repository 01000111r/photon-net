# %%
from p_pack import globals as g

data_folder   = "test-reup-fix"
# data_folder   = "reup-vary-1p"
min_freq    = 0
max_freq      = 2
x_min = 0
x_max = 10
y_min= 0
y_max=0.5
prefix = 'f'
# create two side-by-side plots with independent y-axes
fig, (ax_full, ax_zoom) = g.plt.subplots(1, 2, figsize=(12, 5))

script_dir = g.os.getcwd()  # directory where this script is located
parent_dir   = g.os.path.abspath(g.os.path.join(script_dir, ".."))
data_folder_2 = g.os.path.join(parent_dir, "work", data_folder)

for freq in range(min_freq, max_freq + 1):
    data_name    = f"{prefix}{freq}.npz"
    globals_name = f"{prefix}{freq}g.npz"

    # load
    out      = g.np.load(g.os.path.join(data_folder_2, data_name))
    steps    = out["loss_mem"][:, 0].astype(int)
    losses   = out["loss_mem"][:, 1].astype(float)

    # plot on both axes
    ax_full.plot(steps, losses, linewidth=1.5, label=f"{freq} Hz")
    ax_zoom.plot(steps, losses, linewidth=1.5, label=f"{freq} Hz")

# format the full-range plot
ax_full.set(
    xlabel="step",
    ylabel="loss",
    title="Full Training Curve"
)

# format the zoomed-in plot with its own y-limits
ax_zoom.set(
    xlim=(x_min, x_max),
    ylim=(y_min, y_max),
    xlabel="step",
    ylabel="loss",
    title=f"Zoomed (steps {x_min}–{x_max})"
)

# single legend on the full plot
handles, labels = ax_full.get_legend_handles_labels()
ax_full.legend(handles, labels, loc="upper right", ncol=2, fontsize="small")

# overall title
gl = g.np.load(g.os.path.join(data_folder_2, f"{prefix}{max_freq}g.npz"))
shape = out["carry_0"].shape
fig.suptitle(
    f"Unitary dim: {2*shape[1]}×{2*shape[1]}, p={gl['p_suc_inputs'].item()}, input={gl['input_positions'].item()}"
)

g.plt.tight_layout(rect=[0, 0.03, 1, 0.95])
g.plt.show()


# %%
from p_pack import globals as g
from pathlib import Path

# -------- Configuration --------
data_folder = "init_phases-2"
file_indent = 'k'
indexes_to_plot = [0, 1, 2, 3, 4, 5, 6, 7]   #Choose the indexes you want to plot
legend_names = ["k10", "k10", "k15", "k20", "0.1", "-0.1", "0.01", "-0.01"]  # Custom labels, must match index order
x_min = 1
x_max = 3
y_min = 0.4
y_max = 0.44

# Create two side-by-side plots
fig, (ax_full, ax_zoom) = g.plt.subplots(1, 2, figsize=(12, 5))

# Set data folder path
script_dir = g.os.getcwd()
parent_dir = g.os.path.abspath(g.os.path.join(script_dir, ".."))
data_folder_path = g.os.path.join(parent_dir, "work", data_folder)

# Loop over the selected indexes and legend labels
for idx, label in zip(indexes_to_plot, legend_names):
    # Match file pattern: it{idx}k{var}.npz
    # Load file dynamically by matching prefix
    matches = list(Path(data_folder_path).glob(f"it{idx}{file_indent}*g.npz"))
    if not matches:
        print(f"[Warning] Global file for index {idx} not found.")
        continue

    global_name = matches[0].name
    data_name = global_name.replace("g.npz", ".npz")

    data_path = g.os.path.join(data_folder_path, data_name)
    globals_path = g.os.path.join(data_folder_path, global_name)

    # Load data
    out = g.np.load(data_path)
    steps = out["loss_mem"][:, 0].astype(int)
    losses = out["loss_mem"][:, 1].astype(float)

    ax_full.plot(steps, losses, linewidth=1.5, label=label)
    ax_zoom.plot(steps, losses, linewidth=1.5, label=label)

# Set axis labels and zoom
ax_full.set(
    xlabel="step",
    ylabel="loss",
    title="Full Training Curve"
)

ax_zoom.set(
    xlim=(x_min, x_max),
    ylim=(y_min, y_max),
    xlabel="step",
    ylabel="loss",
    title=f"Zoomed (steps {x_min}–{x_max})"
)

# Legend
handles, labels = ax_full.get_legend_handles_labels()
ax_full.legend(handles, labels, loc="upper right", ncol=2, fontsize="small")

# Use last loaded global file for title
gl = g.np.load(globals_path)
shape = out["carry_0"].shape
fig.suptitle(
    f"Unitary dim: {2*shape[1]}×{2*shape[1]}, p={gl['p_suc_inputs'].item()}, input={gl['input_positions'].item()}"
)

g.plt.tight_layout(rect=[0, 0.03, 1, 0.95])
g.plt.show()


# %%
# ---- Cell 2: init_phases-2 curves with optional test-loss lines ----
from p_pack import globals as g
from pathlib import Path

# -------- Configuration --------
data_folder = "reup-vary-1p-edge/Learning"
file_indent = 'f'
indexes_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Choose the indexes you want to plot
legend_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]  # Custom labels, must match index order
x_min = 0
x_max = 20
y_min = 0.1
y_max = 0.45
plot_test_loss = 0  # set to 1 to show test loss lines
test_loss_folder = "reup-vary-1p-edge/test-1p-edge"

# Create two side-by-side plots
fig, (ax_full, ax_zoom) = g.plt.subplots(1, 2, figsize=(12, 5))

# Set data folder path
script_dir = g.os.getcwd()
parent_dir = g.os.path.abspath(g.os.path.join(script_dir, ".."))
data_folder_path = g.os.path.join(parent_dir, "work", data_folder)
test_folder_path = g.os.path.join(parent_dir, "work", test_loss_folder)

# Loop over the selected indexes and legend labels
for idx, label in zip(indexes_to_plot, legend_names):
    matches = list(Path(data_folder_path).glob(f"it{idx}{file_indent}*g.npz"))
    if not matches:
        print(f"[Warning] Global file for index {idx} not found.")
        continue

    global_name = matches[0].name
    data_name   = global_name.replace("g.npz", ".npz")

    data_path    = g.os.path.join(data_folder_path, data_name)
    globals_path = g.os.path.join(data_folder_path, global_name)

    out    = g.np.load(data_path)
    steps  = out["loss_mem"][:, 0].astype(int)
    losses = out["loss_mem"][:, 1].astype(float)

    # plot training loss and capture its color
    line_full, = ax_full.plot(steps, losses, linewidth=1.5, label=label)
    line_zoom, = ax_zoom.plot(steps, losses, linewidth=1.5, label=label)
    color = line_full.get_color()

    if plot_test_loss:
        test_name = 'tl' + global_name[2:].replace('g.npz', '.npz')
        test_path = g.os.path.join(test_folder_path, test_name)
        if g.os.path.exists(test_path):
            tl = g.np.load(test_path)['test_loss'].item()
            # draw test-loss as a horizontal line in the same color
            ax_full.axline((0, tl), (1, tl),
                           linestyle='--',
                           color=color,
                           label=f'test {label}')
            ax_zoom.axline((0, tl), (1, tl),
                           linestyle='--',
                           color=color)
        else:
            print(f"[Warning] Test loss file {test_name} not found.")

# finalize axes
ax_full.set(xlabel='step', ylabel='loss', title='Full Training Curve')
ax_zoom.set(xlim=(x_min, x_max), ylim=(y_min, y_max),
            xlabel='step', ylabel='loss',
            title=f'Zoomed (steps {x_min}–{x_max})')

handles, labels = ax_full.get_legend_handles_labels()
ax_full.legend(handles, labels, loc='upper right', ncol=2, fontsize='small')

gl    = g.np.load(globals_path)
shape = out['carry_0'].shape
fig.suptitle(
    f"Unitary dim: {2*shape[1]}×{2*shape[1]}, p={gl['p_suc_inputs'].item()}, input={gl['input_positions'].item()}"
)

g.plt.tight_layout(rect=[0, 0.03, 1, 0.95])
g.plt.show()


# %%
# ---- Cell 3: plot test loss vs. index ----
from p_pack import globals as g
from pathlib import Path

test_loss_folder = "reup-vary-1p-edge/test-1p-edge"
file_indent = 't'
indexes_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
legend_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

fig, ax = g.plt.subplots()

script_dir = g.os.getcwd()
parent_dir = g.os.path.abspath(g.os.path.join(script_dir, '..'))
test_folder_path = g.os.path.join(parent_dir, 'work', test_loss_folder)

test_vals = []
for idx, label in zip(indexes_to_plot, legend_names):
    matches = list(Path(test_folder_path).glob(f't{idx}*.npz'))
    if not matches:
        print(f'[Warning] Test loss file for index {idx} not found.')
        continue
    tl = g.np.load(matches[0])['test_loss'].item()
    test_vals.append(tl)
    ax.plot(idx, tl, 'o', label=label)

ax.set(xlabel='index', ylabel='test loss', title='Test Loss vs Index')
ax.legend()
g.plt.show()


# %%
# ---- Cell 2: init_phases-2 curves with optional test-loss lines ----
from p_pack import globals as g
from pathlib import Path

# -------- Configuration --------
data_folder = "edge-photon-reup-vary-s0/Learning"
file_indent = 'f'
indexes_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Choose the indexes you want to plot
legend_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]  # Custom labels, must match index order
x_min = 0
x_max = 20
y_min = 0.1
y_max = 0.45
plot_test_loss = 0  # set to 1 to show test loss lines
test_loss_folder = "reup-vary-1p-edge/test-1p-edge"

# Create two side-by-side plots
fig, (ax_full, ax_zoom) = g.plt.subplots(1, 2, figsize=(12, 5))

# Set data folder path
script_dir = g.os.getcwd()
parent_dir = g.os.path.abspath(g.os.path.join(script_dir, ".."))
data_folder_path = g.os.path.join(parent_dir, "work", data_folder)
test_folder_path = g.os.path.join(parent_dir, "work", test_loss_folder)

# Loop over the selected indexes and legend labels
for idx, label in zip(indexes_to_plot, legend_names):
    matches = list(Path(data_folder_path).glob(f"it{idx}{file_indent}*g.npz"))
    if not matches:
        print(f"[Warning] Global file for index {idx} not found.")
        continue

    global_name = matches[0].name
    data_name   = global_name.replace("g.npz", ".npz")

    data_path    = g.os.path.join(data_folder_path, data_name)
    globals_path = g.os.path.join(data_folder_path, global_name)

    out    = g.np.load(data_path)
    steps  = out["loss_mem"][:, 0].astype(int)
    losses = out["loss_mem"][:, 1].astype(float)

    # plot training loss and capture its color
    line_full, = ax_full.plot(steps, losses, linewidth=1.5, label=label)
    line_zoom, = ax_zoom.plot(steps, losses, linewidth=1.5, label=label)
    color = line_full.get_color()

    if plot_test_loss:
        test_name = 'tl' + global_name[2:].replace('g.npz', '.npz')
        test_path = g.os.path.join(test_folder_path, test_name)
        if g.os.path.exists(test_path):
            tl = g.np.load(test_path)['test_loss'].item()
            # draw test-loss as a horizontal line in the same color
            ax_full.axline((0, tl), (1, tl),
                           linestyle='--',
                           color=color,
                           label=f'test {label}')
            ax_zoom.axline((0, tl), (1, tl),
                           linestyle='--',
                           color=color)
        else:
            print(f"[Warning] Test loss file {test_name} not found.")

# finalize axes
ax_full.set(xlabel='step', ylabel='loss', title='Full Training Curve')
ax_zoom.set(xlim=(x_min, x_max), ylim=(y_min, y_max),
            xlabel='step', ylabel='loss',
            title=f'Zoomed (steps {x_min}–{x_max})')

handles, labels = ax_full.get_legend_handles_labels()
ax_full.legend(handles, labels, loc='upper right', ncol=2, fontsize='small')

gl    = g.np.load(globals_path)
shape = out['carry_0'].shape
fig.suptitle(
    f"Unitary dim: {2*shape[1]}×{2*shape[1]}, p={gl['p_suc_inputs'].item()}, input={gl['input_positions'].item()}, shuffle={gl['shuffle_type'].item()}"
)

g.plt.tight_layout(rect=[0, 0.03, 1, 0.95])
g.plt.show()


# %%
# ---- Cell 3: plot test loss vs. index ----
from p_pack import globals as g
from pathlib import Path

test_loss_folder = "edge-photon-reup-vary-s0/test-1p-edge"
file_indent = 't'
indexes_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
legend_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

fig, ax = g.plt.subplots()

script_dir = g.os.getcwd()
parent_dir = g.os.path.abspath(g.os.path.join(script_dir, '..'))
test_folder_path = g.os.path.join(parent_dir, 'work', test_loss_folder)

test_vals = []
for idx, label in zip(indexes_to_plot, legend_names):
    matches = list(Path(test_folder_path).glob(f't{idx}*.npz'))
    if not matches:
        print(f'[Warning] Test loss file for index {idx} not found.')
        continue
    tl = g.np.load(matches[0])['test_loss'].item()
    test_vals.append(tl)
    ax.plot(idx, tl, 'o', label=label)

ax.set(xlabel='index', ylabel='test loss', title=f'Test Loss vs reup-f-1p-edge (shuffle={gl['shuffle_type'].item()})')
ax.legend()
g.plt.show()


# %%
# ---- Cell 2: init_phases-2 curves with optional test-loss lines ----
from p_pack import globals as g
from pathlib import Path

# -------- Configuration --------
data_folder = "edge-photon-reup-vary-s0-k3/Learning"
file_indent = 'f'
indexes_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Choose the indexes you want to plot
legend_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]  # Custom labels, must match index order
x_min = 0
x_max = 20
y_min = 0.1
y_max = 0.45
plot_test_loss = 0  # set to 1 to show test loss lines
test_loss_folder = "reup-vary-1p-edge/test-1p-edge"

# Create two side-by-side plots
fig, (ax_full, ax_zoom) = g.plt.subplots(1, 2, figsize=(12, 5))

# Set data folder path
script_dir = g.os.getcwd()
parent_dir = g.os.path.abspath(g.os.path.join(script_dir, ".."))
data_folder_path = g.os.path.join(parent_dir, "work", data_folder)
test_folder_path = g.os.path.join(parent_dir, "work", test_loss_folder)

# Loop over the selected indexes and legend labels
for idx, label in zip(indexes_to_plot, legend_names):
    matches = list(Path(data_folder_path).glob(f"it{idx}{file_indent}*g.npz"))
    if not matches:
        print(f"[Warning] Global file for index {idx} not found.")
        continue

    global_name = matches[0].name
    data_name   = global_name.replace("g.npz", ".npz")

    data_path    = g.os.path.join(data_folder_path, data_name)
    globals_path = g.os.path.join(data_folder_path, global_name)

    out    = g.np.load(data_path)
    steps  = out["loss_mem"][:, 0].astype(int)
    losses = out["loss_mem"][:, 1].astype(float)

    # plot training loss and capture its color
    line_full, = ax_full.plot(steps, losses, linewidth=1.5, label=label)
    line_zoom, = ax_zoom.plot(steps, losses, linewidth=1.5, label=label)
    color = line_full.get_color()

    if plot_test_loss:
        test_name = 'tl' + global_name[2:].replace('g.npz', '.npz')
        test_path = g.os.path.join(test_folder_path, test_name)
        if g.os.path.exists(test_path):
            tl = g.np.load(test_path)['test_loss'].item()
            # draw test-loss as a horizontal line in the same color
            ax_full.axline((0, tl), (1, tl),
                           linestyle='--',
                           color=color,
                           label=f'test {label}')
            ax_zoom.axline((0, tl), (1, tl),
                           linestyle='--',
                           color=color)
        else:
            print(f"[Warning] Test loss file {test_name} not found.")

# finalize axes
ax_full.set(xlabel='step', ylabel='loss', title='Full Training Curve')
ax_zoom.set(xlim=(x_min, x_max), ylim=(y_min, y_max),
            xlabel='step', ylabel='loss',
            title=f'Zoomed (steps {x_min}–{x_max})')

handles, labels = ax_full.get_legend_handles_labels()
ax_full.legend(handles, labels, loc='upper right', ncol=2, fontsize='small')

gl    = g.np.load(globals_path)
shape = out['carry_0'].shape
fig.suptitle(
    f"Unitary dim: {2*shape[1]}×{2*shape[1]}, p={gl['p_suc_inputs'].item()}, input={gl['input_positions'].item()}, shuffle={gl['shuffle_type'].item()}"
)

g.plt.tight_layout(rect=[0, 0.03, 1, 0.95])
g.plt.show()


# %%
# ---- Cell 3: plot test loss vs. index ----
from p_pack import globals as g
from pathlib import Path

test_loss_folder = "edge-photon-reup-vary-s0-k3/test-pos0"
file_indent = 't'
indexes_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
legend_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

fig, ax = g.plt.subplots()

script_dir = g.os.getcwd()
parent_dir = g.os.path.abspath(g.os.path.join(script_dir, '..'))
test_folder_path = g.os.path.join(parent_dir, 'work', test_loss_folder)

test_vals = []
for idx, label in zip(indexes_to_plot, legend_names):
    matches = list(Path(test_folder_path).glob(f't{idx}*.npz'))
    if not matches:
        print(f'[Warning] Test loss file for index {idx} not found.')
        continue
    tl = g.np.load(matches[0])['test_loss'].item()
    test_vals.append(tl)
    ax.plot(idx, tl, 'o', label=label)

ax.set(xlabel='index', ylabel='test loss', title=f'Test Loss vs reup-f-1p-edge (shuffle={gl['shuffle_type'].item()}, key={gl['shuffle_key'][1].item()})')
ax.legend()
g.plt.show()


# %%
# ---- Cell 2: init_phases-2 curves with optional test-loss lines ----
from p_pack import globals as g
from pathlib import Path

# -------- Configuration --------
data_folder = "reup-d10-s0/Learning"
file_indent = 'f'
indexes_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Choose the indexes you want to plot
legend_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]  # Custom labels, must match index order
x_min = 0
x_max = 20
y_min = 0.1
y_max = 0.45
plot_test_loss = 0  # set to 1 to show test loss lines
test_loss_folder = "reup-vary-1p-edge/test-1p-edge"

# Create two side-by-side plots
fig, (ax_full, ax_zoom) = g.plt.subplots(1, 2, figsize=(12, 5))

# Set data folder path
script_dir = g.os.getcwd()
parent_dir = g.os.path.abspath(g.os.path.join(script_dir, ".."))
data_folder_path = g.os.path.join(parent_dir, "work", data_folder)
test_folder_path = g.os.path.join(parent_dir, "work", test_loss_folder)

# Loop over the selected indexes and legend labels
for idx, label in zip(indexes_to_plot, legend_names):
    matches = list(Path(data_folder_path).glob(f"it{idx}{file_indent}*g.npz"))
    if not matches:
        print(f"[Warning] Global file for index {idx} not found.")
        continue

    global_name = matches[0].name
    data_name   = global_name.replace("g.npz", ".npz")

    data_path    = g.os.path.join(data_folder_path, data_name)
    globals_path = g.os.path.join(data_folder_path, global_name)

    out    = g.np.load(data_path)
    steps  = out["loss_mem"][:, 0].astype(int)
    losses = out["loss_mem"][:, 1].astype(float)

    # plot training loss and capture its color
    line_full, = ax_full.plot(steps, losses, linewidth=1.5, label=label)
    line_zoom, = ax_zoom.plot(steps, losses, linewidth=1.5, label=label)
    color = line_full.get_color()

    if plot_test_loss:
        test_name = 'tl' + global_name[2:].replace('g.npz', '.npz')
        test_path = g.os.path.join(test_folder_path, test_name)
        if g.os.path.exists(test_path):
            tl = g.np.load(test_path)['test_loss'].item()
            # draw test-loss as a horizontal line in the same color
            ax_full.axline((0, tl), (1, tl),
                           linestyle='--',
                           color=color,
                           label=f'test {label}')
            ax_zoom.axline((0, tl), (1, tl),
                           linestyle='--',
                           color=color)
        else:
            print(f"[Warning] Test loss file {test_name} not found.")

# finalize axes
ax_full.set(xlabel='step', ylabel='loss', title='Full Training Curve')
ax_zoom.set(xlim=(x_min, x_max), ylim=(y_min, y_max),
            xlabel='step', ylabel='loss',
            title=f'Zoomed (steps {x_min}–{x_max})')

handles, labels = ax_full.get_legend_handles_labels()
ax_full.legend(handles, labels, loc='upper right', ncol=2, fontsize='small')

gl    = g.np.load(globals_path)
shape = out['carry_0'].shape
fig.suptitle(
    f"Unitary dim: {2*shape[1]}×{2*shape[1]}, p={gl['p_suc_inputs'].item()}, input={gl['input_positions'].item()}, shuffle={gl['shuffle_type'].item()}"
)

g.plt.tight_layout(rect=[0, 0.03, 1, 0.95])
g.plt.show()


# %%
# ---- Cell 3: plot test loss vs. index ----
from p_pack import globals as g
from pathlib import Path

test_loss_folder = "edge-photon-reup-vary-s0-k52/test-pos0"
file_indent = 't'
indexes_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
legend_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

fig, ax = g.plt.subplots()

script_dir = g.os.getcwd()
parent_dir = g.os.path.abspath(g.os.path.join(script_dir, '..'))
test_folder_path = g.os.path.join(parent_dir, 'work', test_loss_folder)

test_vals = []
for idx, label in zip(indexes_to_plot, legend_names):
    matches = list(Path(test_folder_path).glob(f't{idx}*.npz'))
    if not matches:
        print(f'[Warning] Test loss file for index {idx} not found.')
        continue
    tl = g.np.load(matches[0])['test_loss'].item()
    test_vals.append(tl)
    ax.plot(idx, tl, 'o', label=label)

ax.set(xlabel='index', ylabel='test loss', title=f'Test Loss vs reup-f-1p-edge (shuffle={gl['shuffle_type'].item()}, key={gl['shuffle_key'][1].item()})')
ax.legend()
g.plt.show()


# %%
# ---- Cell 2: init_phases-2 curves with optional test-loss lines ----
from p_pack import globals as g
from pathlib import Path

# -------- Configuration --------
data_folder = "edge-photon-reup-vary-s1/Learning"
file_indent = 'f'
indexes_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Choose the indexes you want to plot
legend_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]  # Custom labels, must match index order
x_min = 0
x_max = 20
y_min = 0.1
y_max = 0.45
plot_test_loss = 0  # set to 1 to show test loss lines
test_loss_folder = "reup-vary-1p-edge/test-1p-edge"

# Create two side-by-side plots
fig, (ax_full, ax_zoom) = g.plt.subplots(1, 2, figsize=(12, 5))

# Set data folder path
script_dir = g.os.getcwd()
parent_dir = g.os.path.abspath(g.os.path.join(script_dir, ".."))
data_folder_path = g.os.path.join(parent_dir, "work", data_folder)
test_folder_path = g.os.path.join(parent_dir, "work", test_loss_folder)

# Loop over the selected indexes and legend labels
for idx, label in zip(indexes_to_plot, legend_names):
    matches = list(Path(data_folder_path).glob(f"it{idx}{file_indent}*g.npz"))
    if not matches:
        print(f"[Warning] Global file for index {idx} not found.")
        continue

    global_name = matches[0].name
    data_name   = global_name.replace("g.npz", ".npz")

    data_path    = g.os.path.join(data_folder_path, data_name)
    globals_path = g.os.path.join(data_folder_path, global_name)

    out    = g.np.load(data_path)
    steps  = out["loss_mem"][:, 0].astype(int)
    losses = out["loss_mem"][:, 1].astype(float)

    # plot training loss and capture its color
    line_full, = ax_full.plot(steps, losses, linewidth=1.5, label=label)
    line_zoom, = ax_zoom.plot(steps, losses, linewidth=1.5, label=label)
    color = line_full.get_color()

    if plot_test_loss:
        test_name = 'tl' + global_name[2:].replace('g.npz', '.npz')
        test_path = g.os.path.join(test_folder_path, test_name)
        if g.os.path.exists(test_path):
            tl = g.np.load(test_path)['test_loss'].item()
            # draw test-loss as a horizontal line in the same color
            ax_full.axline((0, tl), (1, tl),
                           linestyle='--',
                           color=color,
                           label=f'test {label}')
            ax_zoom.axline((0, tl), (1, tl),
                           linestyle='--',
                           color=color)
        else:
            print(f"[Warning] Test loss file {test_name} not found.")

# finalize axes
ax_full.set(xlabel='step', ylabel='loss', title='Full Training Curve')
ax_zoom.set(xlim=(x_min, x_max), ylim=(y_min, y_max),
            xlabel='step', ylabel='loss',
            title=f'Zoomed (steps {x_min}–{x_max})')

handles, labels = ax_full.get_legend_handles_labels()
ax_full.legend(handles, labels, loc='upper right', ncol=2, fontsize='small')

gl    = g.np.load(globals_path)
shape = out['carry_0'].shape
fig.suptitle(
    f"Unitary dim: {2*shape[1]}×{2*shape[1]}, p={gl['p_suc_inputs'].item()}, input={gl['input_positions'].item()}, shuffle={gl['shuffle_type'].item()}"
)

g.plt.tight_layout(rect=[0, 0.03, 1, 0.95])
g.plt.show()


# %%
# ---- Cell 3: plot test loss vs. index ----
from p_pack import globals as g
from pathlib import Path

test_loss_folder = "edge-photon-reup-vary-s1/test-1p-edge"
file_indent = 't'
indexes_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
legend_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

fig, ax = g.plt.subplots()

script_dir = g.os.getcwd()
parent_dir = g.os.path.abspath(g.os.path.join(script_dir, '..'))
test_folder_path = g.os.path.join(parent_dir, 'work', test_loss_folder)

test_vals = []
for idx, label in zip(indexes_to_plot, legend_names):
    matches = list(Path(test_folder_path).glob(f't{idx}*.npz'))
    if not matches:
        print(f'[Warning] Test loss file for index {idx} not found.')
        continue
    tl = g.np.load(matches[0])['test_loss'].item()
    test_vals.append(tl)
    ax.plot(idx, tl, 'o', label=label)

ax.set(xlabel='index', ylabel='test loss', title=f'Test Loss vs reup-f-1p-edge (shuffle={gl['shuffle_type'].item()})')
g.plt.show()


# %%
# ---- Cell 2: init_phases-2 curves with optional test-loss lines ----
from p_pack import globals as g
from pathlib import Path

# -------- Configuration --------
data_folder = "edge-photon-reup-vary-s2/Learning"
file_indent = 'f'
indexes_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Choose the indexes you want to plot
legend_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]  # Custom labels, must match index order
x_min = 0
x_max = 20
y_min = 0.1
y_max = 0.45
plot_test_loss = 0  # set to 1 to show test loss lines
test_loss_folder = "reup-vary-1p-edge/test-1p-edge"

# Create two side-by-side plots
fig, (ax_full, ax_zoom) = g.plt.subplots(1, 2, figsize=(12, 5))

# Set data folder path
script_dir = g.os.getcwd()
parent_dir = g.os.path.abspath(g.os.path.join(script_dir, ".."))
data_folder_path = g.os.path.join(parent_dir, "work", data_folder)
test_folder_path = g.os.path.join(parent_dir, "work", test_loss_folder)

# Loop over the selected indexes and legend labels
for idx, label in zip(indexes_to_plot, legend_names):
    matches = list(Path(data_folder_path).glob(f"it{idx}{file_indent}*g.npz"))
    if not matches:
        print(f"[Warning] Global file for index {idx} not found.")
        continue

    global_name = matches[0].name
    data_name   = global_name.replace("g.npz", ".npz")

    data_path    = g.os.path.join(data_folder_path, data_name)
    globals_path = g.os.path.join(data_folder_path, global_name)

    out    = g.np.load(data_path)
    steps  = out["loss_mem"][:, 0].astype(int)
    losses = out["loss_mem"][:, 1].astype(float)

    # plot training loss and capture its color
    line_full, = ax_full.plot(steps, losses, linewidth=1.5, label=label)
    line_zoom, = ax_zoom.plot(steps, losses, linewidth=1.5, label=label)
    color = line_full.get_color()

    if plot_test_loss:
        test_name = 'tl' + global_name[2:].replace('g.npz', '.npz')
        test_path = g.os.path.join(test_folder_path, test_name)
        if g.os.path.exists(test_path):
            tl = g.np.load(test_path)['test_loss'].item()
            # draw test-loss as a horizontal line in the same color
            ax_full.axline((0, tl), (1, tl),
                           linestyle='--',
                           color=color,
                           label=f'test {label}')
            ax_zoom.axline((0, tl), (1, tl),
                           linestyle='--',
                           color=color)
        else:
            print(f"[Warning] Test loss file {test_name} not found.")

# finalize axes
ax_full.set(xlabel='step', ylabel='loss', title='Full Training Curve')
ax_zoom.set(xlim=(x_min, x_max), ylim=(y_min, y_max),
            xlabel='step', ylabel='loss',
            title=f'Zoomed (steps {x_min}–{x_max})')

handles, labels = ax_full.get_legend_handles_labels()
ax_full.legend(handles, labels, loc='upper right', ncol=2, fontsize='small')

gl    = g.np.load(globals_path)
shape = out['carry_0'].shape
fig.suptitle(
    f"Unitary dim: {2*shape[1]}×{2*shape[1]}, p={gl['p_suc_inputs'].item()}, input={gl['input_positions'].item()}, shuffle={gl['shuffle_type'].item()}"
)

g.plt.tight_layout(rect=[0, 0.03, 1, 0.95])
g.plt.show()



# %%
# ---- Cell 3: plot test loss vs. index ----
from p_pack import globals as g
from pathlib import Path

test_loss_folder = "edge-photon-reup-vary-s2/test-1p-edge"
file_indent = 't'
indexes_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
legend_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

fig, ax = g.plt.subplots()

script_dir = g.os.getcwd()
parent_dir = g.os.path.abspath(g.os.path.join(script_dir, '..'))
test_folder_path = g.os.path.join(parent_dir, 'work', test_loss_folder)

test_vals = []
for idx, label in zip(indexes_to_plot, legend_names):
    matches = list(Path(test_folder_path).glob(f't{idx}*.npz'))
    if not matches:
        print(f'[Warning] Test loss file for index {idx} not found.')
        continue
    tl = g.np.load(matches[0])['test_loss'].item()
    test_vals.append(tl)
    ax.plot(idx, tl, 'o', label=label)

ax.set(xlabel='index', ylabel='test loss', title=f'Test Loss vs reup-f-1p-edge (shuffle={gl['shuffle_type'].item()})')
ax.legend()
g.plt.show()


# %%
# ---- Cell 2: init_phases-2 curves with optional test-loss lines ----
from p_pack import globals as g
from pathlib import Path

# -------- Configuration --------
data_folder = "p1-position0-vs-shuffle-rf2/Learning"
file_indent = 'f'
indexes_to_plot = [0, 1, 2]  # Choose the indexes you want to plot
legend_names = ["0", "1", "2"] # Custom labels, must match index order
x_min = 0
x_max = 20
y_min = 0.1
y_max = 0.45
plot_test_loss = 0 # set to 1 to show test loss lines
test_loss_folder = "p1-position0-vs-shuffle-rf2/test-pos0"

# Create two side-by-side plots
fig, (ax_full, ax_zoom) = g.plt.subplots(1, 2, figsize=(12, 5))

# Set data folder path
script_dir = g.os.getcwd()
parent_dir = g.os.path.abspath(g.os.path.join(script_dir, ".."))
data_folder_path = g.os.path.join(parent_dir, "work", data_folder)
test_folder_path = g.os.path.join(parent_dir, "work", test_loss_folder)

# Loop over the selected indexes and legend labels
for idx, label in zip(indexes_to_plot, legend_names):
    matches = list(Path(data_folder_path).glob(f"it{idx}{file_indent}*g.npz"))
    if not matches:
        print(f"[Warning] Global file for index {idx} not found.")
        continue

    global_name = matches[0].name
    data_name   = global_name.replace("g.npz", ".npz")

    data_path    = g.os.path.join(data_folder_path, data_name)
    globals_path = g.os.path.join(data_folder_path, global_name)

    out    = g.np.load(data_path)
    steps  = out["loss_mem"][:, 0].astype(int)
    losses = out["loss_mem"][:, 1].astype(float)

    # plot training loss and capture its color
    line_full, = ax_full.plot(steps, losses, linewidth=1.5, label=label)
    line_zoom, = ax_zoom.plot(steps, losses, linewidth=1.5, label=label)
    color = line_full.get_color()

    if plot_test_loss:
        test_name = 't' + global_name[2:].replace('g.npz', '.npz')
        test_path = g.os.path.join(test_folder_path, test_name)
        if g.os.path.exists(test_path):
            tl = g.np.load(test_path)['test_loss'].item()
            # draw test-loss as a horizontal line in the same color
            ax_full.axline((0, tl), (1, tl),
                           linestyle='--',
                           color=color,
                           label=f'test {label}')
            ax_zoom.axline((0, tl), (1, tl),
                           linestyle='--',
                           color=color)
        else:
            print(f"[Warning] Test loss file {test_name} not found.")

# finalize axes
ax_full.set(xlabel='step', ylabel='loss', title='Full Training Curve')
ax_zoom.set(xlim=(x_min, x_max), ylim=(y_min, y_max),
            xlabel='step', ylabel='loss',
            title=f'Zoomed (steps {x_min}–{x_max})')

handles, labels = ax_full.get_legend_handles_labels()
ax_full.legend(handles, labels, loc='upper right', ncol=2, fontsize='small')

gl    = g.np.load(globals_path)
shape = out['carry_0'].shape
fig.suptitle(
    f"Unitary dim: {2*shape[1]}×{2*shape[1]}, p={gl['p_suc_inputs'].item()}, input={gl['input_positions'].item()}, shuffle={gl['shuffle_type'].item()}, rf={gl['reupload_freq'].item()}"
)

g.plt.tight_layout(rect=[0, 0.03, 1, 0.95])
g.plt.show()



# %%
# ---- Cell 3: plot test loss vs. index ----
from p_pack import globals as g
from pathlib import Path

test_loss_folder = "p1-position0-vs-shuffle-rf2/test-pos0"
file_indent = 't'
indexes_to_plot = [0, 1, 2]
legend_names = ["0", "1", "2"]

fig, ax = g.plt.subplots()

script_dir = g.os.getcwd()
parent_dir = g.os.path.abspath(g.os.path.join(script_dir, '..'))
test_folder_path = g.os.path.join(parent_dir, 'work', test_loss_folder)

test_vals = []
for idx, label in zip(indexes_to_plot, legend_names):
    matches = list(Path(test_folder_path).glob(f't{idx}*.npz'))
    if not matches:
        print(f'[Warning] Test loss file for index {idx} not found.')
        continue
    tl = g.np.load(matches[0])['test_loss'].item()
    test_vals.append(tl)
    ax.plot(idx, tl, 'o', label=label)

ax.set(xlabel='index', ylabel='test loss', title=f'Test Loss vs reup-f-1p-pos{gl['input_positions'].item()} (shuffle={gl['shuffle_type'].item()}),rf={gl['reupload_freq'].item()}')
ax.legend()
g.plt.show()


# %%
# ---- Cell 2: init_phases-2 curves with optional test-loss lines ----
from p_pack import globals as g
from pathlib import Path

# -------- Configuration --------
data_folder = "p1-position2-vs-shuffle-rf2/Learning"
file_indent = 'f'
indexes_to_plot = [0, 1, 2]  # Choose the indexes you want to plot
legend_names = ["0", "1", "2"] # Custom labels, must match index order
x_min = 0
x_max = 20
y_min = 0.1
y_max = 0.45
plot_test_loss = 0 # set to 1 to show test loss lines
test_loss_folder = "p1-position0-vs-shuffle-rf2/test-pos0"

# Create two side-by-side plots
fig, (ax_full, ax_zoom) = g.plt.subplots(1, 2, figsize=(12, 5))

# Set data folder path
script_dir = g.os.getcwd()
parent_dir = g.os.path.abspath(g.os.path.join(script_dir, ".."))
data_folder_path = g.os.path.join(parent_dir, "work", data_folder)
test_folder_path = g.os.path.join(parent_dir, "work", test_loss_folder)

# Loop over the selected indexes and legend labels
for idx, label in zip(indexes_to_plot, legend_names):
    matches = list(Path(data_folder_path).glob(f"it{idx}{file_indent}*g.npz"))
    if not matches:
        print(f"[Warning] Global file for index {idx} not found.")
        continue

    global_name = matches[0].name
    data_name   = global_name.replace("g.npz", ".npz")

    data_path    = g.os.path.join(data_folder_path, data_name)
    globals_path = g.os.path.join(data_folder_path, global_name)

    out    = g.np.load(data_path)
    steps  = out["loss_mem"][:, 0].astype(int)
    losses = out["loss_mem"][:, 1].astype(float)

    # plot training loss and capture its color
    line_full, = ax_full.plot(steps, losses, linewidth=1.5, label=label)
    line_zoom, = ax_zoom.plot(steps, losses, linewidth=1.5, label=label)
    color = line_full.get_color()

    if plot_test_loss:
        test_name = 't' + global_name[2:].replace('g.npz', '.npz')
        test_path = g.os.path.join(test_folder_path, test_name)
        if g.os.path.exists(test_path):
            tl = g.np.load(test_path)['test_loss'].item()
            # draw test-loss as a horizontal line in the same color
            ax_full.axline((0, tl), (1, tl),
                           linestyle='--',
                           color=color,
                           label=f'test {label}')
            ax_zoom.axline((0, tl), (1, tl),
                           linestyle='--',
                           color=color)
        else:
            print(f"[Warning] Test loss file {test_name} not found.")

# finalize axes
ax_full.set(xlabel='step', ylabel='loss', title='Full Training Curve')
ax_zoom.set(xlim=(x_min, x_max), ylim=(y_min, y_max),
            xlabel='step', ylabel='loss',
            title=f'Zoomed (steps {x_min}–{x_max})')

handles, labels = ax_full.get_legend_handles_labels()
ax_full.legend(handles, labels, loc='upper right', ncol=2, fontsize='small')

gl    = g.np.load(globals_path)
shape = out['carry_0'].shape
fig.suptitle(
    f"Unitary dim: {2*shape[1]}×{2*shape[1]}, p={gl['p_suc_inputs'].item()}, input={gl['input_positions'].item()}, shuffle={gl['shuffle_type'].item()}, rf={gl['reupload_freq'].item()}"
)

g.plt.tight_layout(rect=[0, 0.03, 1, 0.95])
g.plt.show()



# %%
# ---- Cell 3: plot test loss vs. index ----
from p_pack import globals as g
from pathlib import Path

test_loss_folder = "p1-position2-vs-shuffle-rf2/test-pos2"
file_indent = 't'
indexes_to_plot = [0, 1, 2]
legend_names = ["0", "1", "2"]

fig, ax = g.plt.subplots()

script_dir = g.os.getcwd()
parent_dir = g.os.path.abspath(g.os.path.join(script_dir, '..'))
test_folder_path = g.os.path.join(parent_dir, 'work', test_loss_folder)

test_vals = []
for idx, label in zip(indexes_to_plot, legend_names):
    matches = list(Path(test_folder_path).glob(f't{idx}*.npz'))
    if not matches:
        print(f'[Warning] Test loss file for index {idx} not found.')
        continue
    tl = g.np.load(matches[0])['test_loss'].item()
    test_vals.append(tl)
    ax.plot(idx, tl, 'o', label=label)

ax.set(xlabel='index', ylabel='test loss', title=f'Test Loss vs reup-f-1p-pos{gl['input_positions'].item()} (shuffle={gl['shuffle_type'].item()}),rf={gl['reupload_freq'].item()}')
ax.legend()
g.plt.show()


# %%
# ---- Cell 2: init_phases-2 curves with optional test-loss lines ----
from p_pack import globals as g
from pathlib import Path

# -------- Configuration --------
data_folder = "p1-position4-vs-shuffle-rf2/Learning"
file_indent = 'f'
indexes_to_plot = [0, 1, 2]  # Choose the indexes you want to plot
legend_names = ["0", "1", "2"] # Custom labels, must match index order
x_min = 0
x_max = 20
y_min = 0.1
y_max = 0.45
plot_test_loss = 0 # set to 1 to show test loss lines
test_loss_folder = "p1-position0-vs-shuffle-rf2/test-pos0"

# Create two side-by-side plots
fig, (ax_full, ax_zoom) = g.plt.subplots(1, 2, figsize=(12, 5))

# Set data folder path
script_dir = g.os.getcwd()
parent_dir = g.os.path.abspath(g.os.path.join(script_dir, ".."))
data_folder_path = g.os.path.join(parent_dir, "work", data_folder)
test_folder_path = g.os.path.join(parent_dir, "work", test_loss_folder)

# Loop over the selected indexes and legend labels
for idx, label in zip(indexes_to_plot, legend_names):
    matches = list(Path(data_folder_path).glob(f"it{idx}{file_indent}*g.npz"))
    if not matches:
        print(f"[Warning] Global file for index {idx} not found.")
        continue

    global_name = matches[0].name
    data_name   = global_name.replace("g.npz", ".npz")

    data_path    = g.os.path.join(data_folder_path, data_name)
    globals_path = g.os.path.join(data_folder_path, global_name)

    out    = g.np.load(data_path)
    steps  = out["loss_mem"][:, 0].astype(int)
    losses = out["loss_mem"][:, 1].astype(float)

    # plot training loss and capture its color
    line_full, = ax_full.plot(steps, losses, linewidth=1.5, label=label)
    line_zoom, = ax_zoom.plot(steps, losses, linewidth=1.5, label=label)
    color = line_full.get_color()

    if plot_test_loss:
        test_name = 't' + global_name[2:].replace('g.npz', '.npz')
        test_path = g.os.path.join(test_folder_path, test_name)
        if g.os.path.exists(test_path):
            tl = g.np.load(test_path)['test_loss'].item()
            # draw test-loss as a horizontal line in the same color
            ax_full.axline((0, tl), (1, tl),
                           linestyle='--',
                           color=color,
                           label=f'test {label}')
            ax_zoom.axline((0, tl), (1, tl),
                           linestyle='--',
                           color=color)
        else:
            print(f"[Warning] Test loss file {test_name} not found.")

# finalize axes
ax_full.set(xlabel='step', ylabel='loss', title='Full Training Curve')
ax_zoom.set(xlim=(x_min, x_max), ylim=(y_min, y_max),
            xlabel='step', ylabel='loss',
            title=f'Zoomed (steps {x_min}–{x_max})')

handles, labels = ax_full.get_legend_handles_labels()
ax_full.legend(handles, labels, loc='upper right', ncol=2, fontsize='small')

gl    = g.np.load(globals_path)
shape = out['carry_0'].shape
fig.suptitle(
    f"Unitary dim: {2*shape[1]}×{2*shape[1]}, p={gl['p_suc_inputs'].item()}, input={gl['input_positions'].item()}, shuffle={gl['shuffle_type'].item()}, rf={gl['reupload_freq'].item()}"
)

g.plt.tight_layout(rect=[0, 0.03, 1, 0.95])
g.plt.show()



# %%
# ---- Cell 3: plot test loss vs. index ----
from p_pack import globals as g
from pathlib import Path

test_loss_folder = "p1-position4-vs-shuffle-rf2/test-pos4"
file_indent = 't'
indexes_to_plot = [0, 1, 2]
legend_names = ["0", "1", "2"]

fig, ax = g.plt.subplots()

script_dir = g.os.getcwd()
parent_dir = g.os.path.abspath(g.os.path.join(script_dir, '..'))
test_folder_path = g.os.path.join(parent_dir, 'work', test_loss_folder)

test_vals = []
for idx, label in zip(indexes_to_plot, legend_names):
    matches = list(Path(test_folder_path).glob(f't{idx}*.npz'))
    if not matches:
        print(f'[Warning] Test loss file for index {idx} not found.')
        continue
    tl = g.np.load(matches[0])['test_loss'].item()
    test_vals.append(tl)
    ax.plot(idx, tl, 'o', label=label)

ax.set(xlabel='index', ylabel='test loss', title=f'Test Loss vs reup-f-1p-pos{gl['input_positions'].item()} (shuffle={gl['shuffle_type'].item()}),rf={gl['reupload_freq'].item()}')
ax.legend()
g.plt.show()


# %%
# ---- Cell 2: init_phases-2 curves with optional test-loss lines ----
from p_pack import globals as g
from pathlib import Path

# -------- Configuration --------
data_folder = "symmetry-false/Learning"
file_indent = 'f'
indexes_to_plot = [0, 1]  # Choose the indexes you want to plot
legend_names = ["4", "5"] # Custom labels, must match index order
x_min = 0
x_max = 20
y_min = 0.1
y_max = 0.45
plot_test_loss = 0 # set to 1 to show test loss lines
test_loss_folder = "p1-position0-vs-shuffle-rf2/test-pos0"

# Create two side-by-side plots
fig, (ax_full, ax_zoom) = g.plt.subplots(1, 2, figsize=(12, 5))

# Set data folder path
script_dir = g.os.getcwd()
parent_dir = g.os.path.abspath(g.os.path.join(script_dir, ".."))
data_folder_path = g.os.path.join(parent_dir, "work", data_folder)
test_folder_path = g.os.path.join(parent_dir, "work", test_loss_folder)

# Loop over the selected indexes and legend labels
for idx, label in zip(indexes_to_plot, legend_names):
    matches = list(Path(data_folder_path).glob(f"it{idx}{file_indent}*g.npz"))
    if not matches:
        print(f"[Warning] Global file for index {idx} not found.")
        continue

    global_name = matches[0].name
    data_name   = global_name.replace("g.npz", ".npz")

    data_path    = g.os.path.join(data_folder_path, data_name)
    globals_path = g.os.path.join(data_folder_path, global_name)

    out    = g.np.load(data_path)
    steps  = out["loss_mem"][:, 0].astype(int)
    losses = out["loss_mem"][:, 1].astype(float)

    # plot training loss and capture its color
    line_full, = ax_full.plot(steps, losses, linewidth=1.5, label=label)
    line_zoom, = ax_zoom.plot(steps, losses, linewidth=1.5, label=label)
    color = line_full.get_color()

    if plot_test_loss:
        test_name = 't' + global_name[2:].replace('g.npz', '.npz')
        test_path = g.os.path.join(test_folder_path, test_name)
        if g.os.path.exists(test_path):
            tl = g.np.load(test_path)['test_loss'].item()
            # draw test-loss as a horizontal line in the same color
            ax_full.axline((0, tl), (1, tl),
                           linestyle='--',
                           color=color,
                           label=f'test {label}')
            ax_zoom.axline((0, tl), (1, tl),
                           linestyle='--',
                           color=color)
        else:
            print(f"[Warning] Test loss file {test_name} not found.")

# finalize axes
ax_full.set(xlabel='step', ylabel='loss', title='Full Training Curve')
ax_zoom.set(xlim=(x_min, x_max), ylim=(y_min, y_max),
            xlabel='step', ylabel='loss',
            title=f'Zoomed (steps {x_min}–{x_max})')

handles, labels = ax_full.get_legend_handles_labels()
ax_full.legend(handles, labels, loc='upper right', ncol=2, fontsize='small')

gl    = g.np.load(globals_path)
shape = out['carry_0'].shape
fig.suptitle(
    f"Unitary dim: {2*shape[1]}×{2*shape[1]}, p={gl['p_suc_inputs'].item()}, input={gl['input_positions'].item()}, shuffle={gl['shuffle_type'].item()}, rf={gl['reupload_freq'].item()}"
)

g.plt.tight_layout(rect=[0, 0.03, 1, 0.95])
g.plt.show()



# %%
# ---- Cell 2: init_phases-2 curves with optional test-loss lines ----
from p_pack import globals as g
from pathlib import Path

# -------- Configuration --------
data_folder = "symmetry-true-2p/Learning"
file_indent = 'f'
indexes_to_plot = [0, 1, 2]  # Choose the indexes you want to plot
legend_names = ["1", "2", "3"] # Custom labels, must match index order
x_min = 0
x_max = 20
y_min = 0.1
y_max = 0.45
plot_test_loss = 0 # set to 1 to show test loss lines
test_loss_folder = "p1-position0-vs-shuffle-rf2/test-pos0"

# Create two side-by-side plots
fig, (ax_full, ax_zoom) = g.plt.subplots(1, 2, figsize=(12, 5))

# Set data folder path
script_dir = g.os.getcwd()
parent_dir = g.os.path.abspath(g.os.path.join(script_dir, ".."))
data_folder_path = g.os.path.join(parent_dir, "work", data_folder)
test_folder_path = g.os.path.join(parent_dir, "work", test_loss_folder)

# Loop over the selected indexes and legend labels
for idx, label in zip(indexes_to_plot, legend_names):
    matches = list(Path(data_folder_path).glob(f"it{idx}{file_indent}*g.npz"))
    if not matches:
        print(f"[Warning] Global file for index {idx} not found.")
        continue

    global_name = matches[0].name
    data_name   = global_name.replace("g.npz", ".npz")

    data_path    = g.os.path.join(data_folder_path, data_name)
    globals_path = g.os.path.join(data_folder_path, global_name)

    out    = g.np.load(data_path)
    steps  = out["loss_mem"][:, 0].astype(int)
    losses = out["loss_mem"][:, 1].astype(float)

    # plot training loss and capture its color
    line_full, = ax_full.plot(steps, losses, linewidth=1.5, label=label)
    line_zoom, = ax_zoom.plot(steps, losses, linewidth=1.5, label=label)
    color = line_full.get_color()

    if plot_test_loss:
        test_name = 't' + global_name[2:].replace('g.npz', '.npz')
        test_path = g.os.path.join(test_folder_path, test_name)
        if g.os.path.exists(test_path):
            tl = g.np.load(test_path)['test_loss'].item()
            # draw test-loss as a horizontal line in the same color
            ax_full.axline((0, tl), (1, tl),
                           linestyle='--',
                           color=color,
                           label=f'test {label}')
            ax_zoom.axline((0, tl), (1, tl),
                           linestyle='--',
                           color=color)
        else:
            print(f"[Warning] Test loss file {test_name} not found.")

# finalize axes
ax_full.set(xlabel='step', ylabel='loss', title='Full Training Curve')
ax_zoom.set(xlim=(x_min, x_max), ylim=(y_min, y_max),
            xlabel='step', ylabel='loss',
            title=f'Zoomed (steps {x_min}–{x_max})')

handles, labels = ax_full.get_legend_handles_labels()
ax_full.legend(handles, labels, loc='upper right', ncol=2, fontsize='small')

gl    = g.np.load(globals_path)
shape = out['carry_0'].shape
fig.suptitle(
    f"Unitary dim: {2*shape[1]}×{2*shape[1]}, p={gl['p_suc_inputs'].item()}, input=2 shuffle={gl['shuffle_type'].item()}, rf={gl['reupload_freq'].item()}"
)

g.plt.tight_layout(rect=[0, 0.03, 1, 0.95])
g.plt.show()



# %%
# ---- Cell 3: plot test loss vs. index ----
from p_pack import globals as g
from pathlib import Path

test_loss_folder = "symmetry-true-2p/test-p2"
file_indent = 't'
indexes_to_plot = [0, 1, 2]
legend_names = ["0", "1", "2"]

fig, ax = g.plt.subplots()

script_dir = g.os.getcwd()
parent_dir = g.os.path.abspath(g.os.path.join(script_dir, '..'))
test_folder_path = g.os.path.join(parent_dir, 'work', test_loss_folder)

test_vals = []
for idx, label in zip(indexes_to_plot, legend_names):
    matches = list(Path(test_folder_path).glob(f't{idx}*.npz'))
    if not matches:
        print(f'[Warning] Test loss file for index {idx} not found.')
        continue
    tl = g.np.load(matches[0])['test_loss'].item()
    test_vals.append(tl)
    ax.plot(idx, tl, 'o', label=label)

ax.set(xlabel='index', ylabel='test loss', title=f'Test Loss vs reup-f-1p-pos2 (shuffle={gl['shuffle_type'].item()}),rf={gl['reupload_freq'].item()}')
ax.legend()
g.plt.show()


# %%
# ---- Cell 2: init_phases-2 curves with optional test-loss lines ----
from p_pack import globals as g
from pathlib import Path

# -------- Configuration --------
data_folder = "symmetry-false-2p/Learning"
file_indent = 'f'
indexes_to_plot = [0, 1, 2]  # Choose the indexes you want to plot
legend_names = ["0", "1", "2"] # Custom labels, must match index order
x_min = 0
x_max = 20
y_min = 0.1
y_max = 0.45
plot_test_loss = 0 # set to 1 to show test loss lines
test_loss_folder = "p1-position0-vs-shuffle-rf2/test-pos0"

# Create two side-by-side plots
fig, (ax_full, ax_zoom) = g.plt.subplots(1, 2, figsize=(12, 5))

# Set data folder path
script_dir = g.os.getcwd()
parent_dir = g.os.path.abspath(g.os.path.join(script_dir, ".."))
data_folder_path = g.os.path.join(parent_dir, "work", data_folder)
test_folder_path = g.os.path.join(parent_dir, "work", test_loss_folder)

# Loop over the selected indexes and legend labels
for idx, label in zip(indexes_to_plot, legend_names):
    matches = list(Path(data_folder_path).glob(f"it{idx}{file_indent}*g.npz"))
    if not matches:
        print(f"[Warning] Global file for index {idx} not found.")
        continue

    global_name = matches[0].name
    data_name   = global_name.replace("g.npz", ".npz")

    data_path    = g.os.path.join(data_folder_path, data_name)
    globals_path = g.os.path.join(data_folder_path, global_name)

    out    = g.np.load(data_path)
    steps  = out["loss_mem"][:, 0].astype(int)
    losses = out["loss_mem"][:, 1].astype(float)

    # plot training loss and capture its color
    line_full, = ax_full.plot(steps, losses, linewidth=1.5, label=label)
    line_zoom, = ax_zoom.plot(steps, losses, linewidth=1.5, label=label)
    color = line_full.get_color()

    if plot_test_loss:
        test_name = 't' + global_name[2:].replace('g.npz', '.npz')
        test_path = g.os.path.join(test_folder_path, test_name)
        if g.os.path.exists(test_path):
            tl = g.np.load(test_path)['test_loss'].item()
            # draw test-loss as a horizontal line in the same color
            ax_full.axline((0, tl), (1, tl),
                           linestyle='--',
                           color=color,
                           label=f'test {label}')
            ax_zoom.axline((0, tl), (1, tl),
                           linestyle='--',
                           color=color)
        else:
            print(f"[Warning] Test loss file {test_name} not found.")

# finalize axes
ax_full.set(xlabel='step', ylabel='loss', title='Full Training Curve')
ax_zoom.set(xlim=(x_min, x_max), ylim=(y_min, y_max),
            xlabel='step', ylabel='loss',
            title=f'Zoomed (steps {x_min}–{x_max})')

handles, labels = ax_full.get_legend_handles_labels()
ax_full.legend(handles, labels, loc='upper right', ncol=2, fontsize='small')

gl    = g.np.load(globals_path)
shape = out['carry_0'].shape
fig.suptitle(
    f"Unitary dim: {2*shape[1]}×{2*shape[1]}, p={gl['p_suc_inputs'].item()}, input=2 shuffle={gl['shuffle_type'].item()}, rf={gl['reupload_freq'].item()}"
)

g.plt.tight_layout(rect=[0, 0.03, 1, 0.95])
g.plt.show()



# %%


# %%


# %%
# ---- Cell 3: plot test loss vs. index ----
from p_pack import globals as g
from pathlib import Path

test_loss_folder = "symmetry-false-2p/test-p2"
file_indent = 't'
indexes_to_plot = [0, 1, 2]
legend_names = ["0", "1", "2"]

fig, ax = g.plt.subplots()

script_dir = g.os.getcwd()
parent_dir = g.os.path.abspath(g.os.path.join(script_dir, '..'))
test_folder_path = g.os.path.join(parent_dir, 'work', test_loss_folder)

test_vals = []
for idx, label in zip(indexes_to_plot, legend_names):
    matches = list(Path(test_folder_path).glob(f't{idx}*.npz'))
    if not matches:
        print(f'[Warning] Test loss file for index {idx} not found.')
        continue
    tl = g.np.load(matches[0])['test_loss'].item()
    test_vals.append(tl)
    ax.plot(idx, tl, 'o', label=label)

ax.set(xlabel='index', ylabel='test loss', title=f'Test Loss vs reup-f-1p-pos2 (shuffle={gl['shuffle_type'].item()}),rf={gl['reupload_freq'].item()}')
ax.legend()
g.plt.show()


# %%
# ---- Cell 2: init_phases-2 curves with optional test-loss lines ----
from p_pack import globals as g
from pathlib import Path

# -------- Configuration --------
data_folders = ("reup-d4-s0/Learning", "reup-d4-s1/Learning", "reup-d4-s2/Learning")
for i in data_folders:
    data_folder = i
    file_indent = 'f'
    indexes_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Choose the indexes you want to plot
    legend_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]  # Custom labels, must match index order
    x_min = 0
    x_max = 20
    y_min = 0.1
    y_max = 0.45
    plot_test_loss = 0 # set to 1 to show test loss lines
    test_loss_folder = "p1-position0-vs-shuffle-rf2/test-pos0"

    # Create two side-by-side plots
    fig, (ax_full, ax_zoom) = g.plt.subplots(1, 2, figsize=(12, 5))

    # Set data folder path
    script_dir = g.os.getcwd()
    parent_dir = g.os.path.abspath(g.os.path.join(script_dir, ".."))
    data_folder_path = g.os.path.join(parent_dir, "work", data_folder)
    test_folder_path = g.os.path.join(parent_dir, "work", test_loss_folder)

    # Loop over the selected indexes and legend labels
    for idx, label in zip(indexes_to_plot, legend_names):
        matches = list(Path(data_folder_path).glob(f"it{idx}{file_indent}*g.npz"))
        if not matches:
            print(f"[Warning] Global file for index {idx} not found.")
            continue

        global_name = matches[0].name
        data_name   = global_name.replace("g.npz", ".npz")

        data_path    = g.os.path.join(data_folder_path, data_name)
        globals_path = g.os.path.join(data_folder_path, global_name)

        out    = g.np.load(data_path)
        steps  = out["loss_mem"][:, 0].astype(int)
        losses = out["loss_mem"][:, 1].astype(float)

        # plot training loss and capture its color
        line_full, = ax_full.plot(steps, losses, linewidth=1.5, label=label)
        line_zoom, = ax_zoom.plot(steps, losses, linewidth=1.5, label=label)
        color = line_full.get_color()

        if plot_test_loss:
            test_name = 't' + global_name[2:].replace('g.npz', '.npz')
            test_path = g.os.path.join(test_folder_path, test_name)
            if g.os.path.exists(test_path):
                tl = g.np.load(test_path)['test_loss'].item()
                # draw test-loss as a horizontal line in the same color
                ax_full.axline((0, tl), (1, tl),
                            linestyle='--',
                            color=color,
                            label=f'test {label}')
                ax_zoom.axline((0, tl), (1, tl),
                            linestyle='--',
                            color=color)
            else:
                print(f"[Warning] Test loss file {test_name} not found.")

    # finalize axes
    ax_full.set(xlabel='step', ylabel='loss', title='Full Training Curve')
    ax_zoom.set(xlim=(x_min, x_max), ylim=(y_min, y_max),
                xlabel='step', ylabel='loss',
                title=f'Zoomed (steps {x_min}–{x_max})')

    handles, labels = ax_full.get_legend_handles_labels()
    ax_full.legend(handles, labels, loc='upper right', ncol=2, fontsize='small')

    gl    = g.np.load(globals_path)
    shape = out['carry_0'].shape
    fig.suptitle(
        f"Unitary dim: {2*shape[1]}×{2*shape[1]}, p={gl['p_suc_inputs'].item()}, input=2 shuffle={gl['shuffle_type'].item()}, rf={gl['reupload_freq'].item()}"
    )

    g.plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    g.plt.show()



# %%
# ---- Cell 3: plot test loss vs. index ----
from p_pack import globals as g
from pathlib import Path

data_folders = ("reup-d4-s0/test-p1", "reup-d4-s1/test-p1", "reup-d4-s2/test-p1")
for i in data_folders:
    test_loss_folder= i
    file_indent = 't'
    indexes_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    legend_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    fig, ax = g.plt.subplots()

    script_dir = g.os.getcwd()
    parent_dir = g.os.path.abspath(g.os.path.join(script_dir, '..'))
    test_folder_path = g.os.path.join(parent_dir, 'work', test_loss_folder)

    test_vals = []
    for idx, label in zip(indexes_to_plot, legend_names):
        matches = list(Path(test_folder_path).glob(f't{idx}*.npz'))
        if not matches:
            print(f'[Warning] Test loss file for index {idx} not found.')
            continue
        tl = g.np.load(matches[0])['test_loss'].item()
        test_vals.append(tl)
        ax.plot(idx, tl, 'o', label=label)

    ax.set(xlabel='index', ylabel='test loss', title=f'Test Loss vs reup-f-1p-edge (shuffle={gl['shuffle_type'].item()})')
    ax.legend()
    g.plt.show()

# %%
# ---- Cell 2: init_phases-2 curves with optional test-loss lines ----
from p_pack import globals as g
from pathlib import Path

# -------- Configuration --------
data_folders = ("reup-d6-s0/Learning", "reup-d6-s1/Learning", "reup-d6-s2/Learning")
for i in data_folders:
    data_folder = i
    file_indent = 'f'
    indexes_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Choose the indexes you want to plot
    legend_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]  # Custom labels, must match index order
    x_min = 0
    x_max = 20
    y_min = 0.1
    y_max = 0.45
    plot_test_loss = 0 # set to 1 to show test loss lines
    test_loss_folder = "p1-position0-vs-shuffle-rf2/test-pos0"

    # Create two side-by-side plots
    fig, (ax_full, ax_zoom) = g.plt.subplots(1, 2, figsize=(12, 5))

    # Set data folder path
    script_dir = g.os.getcwd()
    parent_dir = g.os.path.abspath(g.os.path.join(script_dir, ".."))
    data_folder_path = g.os.path.join(parent_dir, "work", data_folder)
    test_folder_path = g.os.path.join(parent_dir, "work", test_loss_folder)

    # Loop over the selected indexes and legend labels
    for idx, label in zip(indexes_to_plot, legend_names):
        matches = list(Path(data_folder_path).glob(f"it{idx}{file_indent}*g.npz"))
        if not matches:
            print(f"[Warning] Global file for index {idx} not found.")
            continue

        global_name = matches[0].name
        data_name   = global_name.replace("g.npz", ".npz")

        data_path    = g.os.path.join(data_folder_path, data_name)
        globals_path = g.os.path.join(data_folder_path, global_name)

        out    = g.np.load(data_path)
        steps  = out["loss_mem"][:, 0].astype(int)
        losses = out["loss_mem"][:, 1].astype(float)

        # plot training loss and capture its color
        line_full, = ax_full.plot(steps, losses, linewidth=1.5, label=label)
        line_zoom, = ax_zoom.plot(steps, losses, linewidth=1.5, label=label)
        color = line_full.get_color()

        if plot_test_loss:
            test_name = 't' + global_name[2:].replace('g.npz', '.npz')
            test_path = g.os.path.join(test_folder_path, test_name)
            if g.os.path.exists(test_path):
                tl = g.np.load(test_path)['test_loss'].item()
                # draw test-loss as a horizontal line in the same color
                ax_full.axline((0, tl), (1, tl),
                            linestyle='--',
                            color=color,
                            label=f'test {label}')
                ax_zoom.axline((0, tl), (1, tl),
                            linestyle='--',
                            color=color)
            else:
                print(f"[Warning] Test loss file {test_name} not found.")

    # finalize axes
    ax_full.set(xlabel='step', ylabel='loss', title='Full Training Curve')
    ax_zoom.set(xlim=(x_min, x_max), ylim=(y_min, y_max),
                xlabel='step', ylabel='loss',
                title=f'Zoomed (steps {x_min}–{x_max})')

    handles, labels = ax_full.get_legend_handles_labels()
    ax_full.legend(handles, labels, loc='upper right', ncol=2, fontsize='small')

    gl    = g.np.load(globals_path)
    shape = out['carry_0'].shape
    fig.suptitle(
        f"Unitary dim: {2*shape[1]}×{2*shape[1]}, p={gl['p_suc_inputs'].item()}, input=2 shuffle={gl['shuffle_type'].item()}, rf={gl['reupload_freq'].item()}"
    )

    g.plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    g.plt.show()



# %%
# ---- Cell 3: plot test loss vs. index ----
from p_pack import globals as g
from pathlib import Path

data_folders = ("reup-d6-s0/test-p1", "reup-d6-s1/test-p1", "reup-d6-s2/test-p1")
for i in data_folders:
    test_loss_folder= i
    file_indent = 't'
    indexes_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    legend_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    fig, ax = g.plt.subplots()

    script_dir = g.os.getcwd()
    parent_dir = g.os.path.abspath(g.os.path.join(script_dir, '..'))
    test_folder_path = g.os.path.join(parent_dir, 'work', test_loss_folder)

    test_vals = []
    for idx, label in zip(indexes_to_plot, legend_names):
        matches = list(Path(test_folder_path).glob(f't{idx}*.npz'))
        if not matches:
            print(f'[Warning] Test loss file for index {idx} not found.')
            continue
        tl = g.np.load(matches[0])['test_loss'].item()
        test_vals.append(tl)
        ax.plot(idx, tl, 'o', label=label)

    ax.set(xlabel='index', ylabel='test loss', title=f'Test Loss vs reup-f-1p-edge (shuffle={gl['shuffle_type'].item()})')
    ax.legend()
    g.plt.show()

# %%
# ---- Cell 2: init_phases-2 curves with optional test-loss lines ----
from p_pack import globals as g
from pathlib import Path

# -------- Configuration --------
data_folders = ("reup-d7-s0/Learning", "reup-d7-s1/Learning", "reup-d7-s2/Learning")
for i in data_folders:
    data_folder = i
    file_indent = 'f'
    indexes_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Choose the indexes you want to plot
    legend_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]  # Custom labels, must match index order
    x_min = 0
    x_max = 20
    y_min = 0.1
    y_max = 0.45
    plot_test_loss = 0 # set to 1 to show test loss lines
    test_loss_folder = "p1-position0-vs-shuffle-rf2/test-pos0"

    # Create two side-by-side plots
    fig, (ax_full, ax_zoom) = g.plt.subplots(1, 2, figsize=(12, 5))

    # Set data folder path
    script_dir = g.os.getcwd()
    parent_dir = g.os.path.abspath(g.os.path.join(script_dir, ".."))
    data_folder_path = g.os.path.join(parent_dir, "work", data_folder)
    test_folder_path = g.os.path.join(parent_dir, "work", test_loss_folder)

    # Loop over the selected indexes and legend labels
    for idx, label in zip(indexes_to_plot, legend_names):
        matches = list(Path(data_folder_path).glob(f"it{idx}{file_indent}*g.npz"))
        if not matches:
            print(f"[Warning] Global file for index {idx} not found.")
            continue

        global_name = matches[0].name
        data_name   = global_name.replace("g.npz", ".npz")

        data_path    = g.os.path.join(data_folder_path, data_name)
        globals_path = g.os.path.join(data_folder_path, global_name)

        out    = g.np.load(data_path)
        steps  = out["loss_mem"][:, 0].astype(int)
        losses = out["loss_mem"][:, 1].astype(float)

        # plot training loss and capture its color
        line_full, = ax_full.plot(steps, losses, linewidth=1.5, label=label)
        line_zoom, = ax_zoom.plot(steps, losses, linewidth=1.5, label=label)
        color = line_full.get_color()

        if plot_test_loss:
            test_name = 't' + global_name[2:].replace('g.npz', '.npz')
            test_path = g.os.path.join(test_folder_path, test_name)
            if g.os.path.exists(test_path):
                tl = g.np.load(test_path)['test_loss'].item()
                # draw test-loss as a horizontal line in the same color
                ax_full.axline((0, tl), (1, tl),
                            linestyle='--',
                            color=color,
                            label=f'test {label}')
                ax_zoom.axline((0, tl), (1, tl),
                            linestyle='--',
                            color=color)
            else:
                print(f"[Warning] Test loss file {test_name} not found.")

    # finalize axes
    ax_full.set(xlabel='step', ylabel='loss', title='Full Training Curve')
    ax_zoom.set(xlim=(x_min, x_max), ylim=(y_min, y_max),
                xlabel='step', ylabel='loss',
                title=f'Zoomed (steps {x_min}–{x_max})')

    handles, labels = ax_full.get_legend_handles_labels()
    ax_full.legend(handles, labels, loc='upper right', ncol=2, fontsize='small')

    gl    = g.np.load(globals_path)
    shape = out['carry_0'].shape
    fig.suptitle(
        f"Unitary dim: {2*shape[1]}×{2*shape[1]}, p={gl['p_suc_inputs'].item()}, input=2 shuffle={gl['shuffle_type'].item()}, rf={gl['reupload_freq'].item()}"
    )

    g.plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    g.plt.show()



# %%
# ---- Cell 3: plot test loss vs. index ----
from p_pack import globals as g
from pathlib import Path

data_folders = ("reup-d7-s0/test-p1", "reup-d7-s1/test-p1", "reup-d7-s2/test-p1")
for i in data_folders:
    test_loss_folder= i
    file_indent = 't'
    indexes_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    legend_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    fig, ax = g.plt.subplots()

    script_dir = g.os.getcwd()
    parent_dir = g.os.path.abspath(g.os.path.join(script_dir, '..'))
    test_folder_path = g.os.path.join(parent_dir, 'work', test_loss_folder)

    test_vals = []
    for idx, label in zip(indexes_to_plot, legend_names):
        matches = list(Path(test_folder_path).glob(f't{idx}*.npz'))
        if not matches:
            print(f'[Warning] Test loss file for index {idx} not found.')
            continue
        tl = g.np.load(matches[0])['test_loss'].item()
        test_vals.append(tl)
        ax.plot(idx, tl, 'o', label=label)

    ax.set(xlabel='index', ylabel='test loss', title=f'Test Loss vs reup-f-1p-edge (shuffle={gl['shuffle_type'].item()})')
    ax.legend()
    g.plt.show()

# %%
# ---- Cell 3: plot test loss vs. index ----
from p_pack import globals as g
from pathlib import Path

test_loss_folder = "new-test-test/test-acc-500"
file_indent = 't'
indexes_to_plot = [0, 1, 2, 3, 4]
legend_names = ["0", "1", "2", "3", "4"]

fig, ax = g.plt.subplots()

script_dir = g.os.getcwd()
parent_dir = g.os.path.abspath(g.os.path.join(script_dir, '..'))
test_folder_path = g.os.path.join(parent_dir, 'work', test_loss_folder)

test_vals = []
for idx, label in zip(indexes_to_plot, legend_names):
    matches = list(Path(test_folder_path).glob(f't{idx}*.npz'))
    if not matches:
        print(f'[Warning] Test loss file for index {idx} not found.')
        continue
    tl = g.np.load(matches[0])['test_loss'].item()
    test_vals.append(tl)
    ax.plot(idx, tl, 'o', label=label)

ax.set(xlabel='index', ylabel='test loss', title=f'Test Loss vs reup-f-1p-edge (shuffle={gl['shuffle_type'].item()})')
ax.legend()
g.plt.show()


# %%
# ---- Cell 2: init_phases-2 curves with optional test-loss lines ----
from p_pack import globals as g
from pathlib import Path

# -------- Configuration --------
data_folders = ("p_num_long_runs/Learning",)
for i in data_folders:
    data_folder = i
    file_indent = 'p'
    indexes_to_plot = [1,2,3]  # Choose the indexes you want to plot
    legend_names = ["1", "2", "3"]  # Custom labels, must match index order
    x_min = 280
    x_max = 300
    y_min = 0.05
    y_max = 0.15
    plot_test_loss = 0 # set to 1 to show test loss lines
    test_loss_folder = "p1-position0-vs-shuffle-rf2/test-pos0"

    # Create two side-by-side plots
    fig, (ax_full, ax_zoom) = g.plt.subplots(1, 2, figsize=(12, 5))

    # Set data folder path
    script_dir = g.os.getcwd()
    parent_dir = g.os.path.abspath(g.os.path.join(script_dir, ".."))
    data_folder_path = g.os.path.join(parent_dir, "work", data_folder)
    test_folder_path = g.os.path.join(parent_dir, "work", test_loss_folder)

    # Loop over the selected indexes and legend labels
    for idx, label in zip(indexes_to_plot, legend_names):
        matches = list(Path(data_folder_path).glob(f"it{idx}{file_indent}*g.npz"))
        if not matches:
            print(f"[Warning] Global file for index {idx} not found.")
            continue

        global_name = matches[0].name
        data_name   = global_name.replace("g.npz", ".npz")

        data_path    = g.os.path.join(data_folder_path, data_name)
        globals_path = g.os.path.join(data_folder_path, global_name)

        out    = g.np.load(data_path)
        steps  = out["loss_mem"][:, 0].astype(int)
        losses = out["loss_mem"][:, 1].astype(float)

        # plot training loss and capture its color
        line_full, = ax_full.plot(steps, losses, linewidth=1.5, label=label)
        line_zoom, = ax_zoom.plot(steps, losses, linewidth=1.5, label=label)
        color = line_full.get_color()

        if plot_test_loss:
            test_name = 't' + global_name[2:].replace('g.npz', '.npz')
            test_path = g.os.path.join(test_folder_path, test_name)
            if g.os.path.exists(test_path):
                tl = g.np.load(test_path)['test_loss'].item()
                # draw test-loss as a horizontal line in the same color
                ax_full.axline((0, tl), (1, tl),
                            linestyle='--',
                            color=color,
                            label=f'test {label}')
                ax_zoom.axline((0, tl), (1, tl),
                            linestyle='--',
                            color=color)
            else:
                print(f"[Warning] Test loss file {test_name} not found.")

    # finalize axes
    ax_full.set(xlabel='step', ylabel='loss', title='Full Training Curve')
    ax_zoom.set(xlim=(x_min, x_max), ylim=(y_min, y_max),
                xlabel='step', ylabel='loss',
                title=f'Zoomed (steps {x_min}–{x_max})')

    handles, labels = ax_full.get_legend_handles_labels()
    ax_full.legend(handles, labels, loc='upper right', ncol=2, fontsize='small')

    gl    = g.np.load(globals_path)
    shape = out['carry_0'].shape
    fig.suptitle(
       # f"Unitary dim: {2*shape[1]}×{2*shape[1]}, p={gl['p_suc_inputs'].item()}, input=2 shuffle={gl['shuffle_type'].item()}, rf={gl['reupload_freq'].item()}"
       "10x10 shuffle 0, rf 4, varying photon number "
    )

    g.plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    g.plt.show()



# %%
# ---- Cell 3: plot test loss vs. index ----
from p_pack import globals as g
from pathlib import Path

test_loss_folder = "p3-reup-vary/test-acc-500"
file_indent = 't'
indexes_to_plot = [0, 1, 2, 3, 4,5,6,7,8,9]
legend_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

fig, ax = g.plt.subplots()

script_dir = g.os.getcwd()
parent_dir = g.os.path.abspath(g.os.path.join(script_dir, '..'))
test_folder_path = g.os.path.join(parent_dir, 'work', test_loss_folder)

test_vals = []
for idx, label in zip(indexes_to_plot, legend_names):
    matches = list(Path(test_folder_path).glob(f't{idx}*.npz'))
    if not matches:
        print(f'[Warning] Test loss file for index {idx} not found.')
        continue
    tl = g.np.load(matches[0])['test_loss'].item()
    test_vals.append(tl)
    ax.plot(idx, tl, 'o', label=label)

ax.set(xlabel='index', ylabel='test loss', title=f'Test Loss vs reup-f-3p(shuffle={gl['shuffle_type'].item()})')
ax.legend()
g.plt.show()


# %%
# ---- Cell 3: plot test loss vs. index ----
from p_pack import globals as g
from pathlib import Path

test_loss_folder = "p3-reup-vary/test-acc2-500"
file_indent = 't'
indexes_to_plot = [0, 1, 2, 3, 4,5,6,7,8,9]
legend_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

fig, ax = g.plt.subplots()

script_dir = g.os.getcwd()
parent_dir = g.os.path.abspath(g.os.path.join(script_dir, '..'))
test_folder_path = g.os.path.join(parent_dir, 'work', test_loss_folder)

test_vals = []
for idx, label in zip(indexes_to_plot, legend_names):
    matches = list(Path(test_folder_path).glob(f't{idx}*.npz'))
    if not matches:
        print(f'[Warning] Test loss file for index {idx} not found.')
        continue
    tl = g.np.load(matches[0])['test_loss'].item()
    test_vals.append(tl)
    ax.plot(idx, tl, 'o', label=label)

ax.set(xlabel='index', ylabel='test loss', title=f'Test Loss vs reup-f-3p(shuffle={gl['shuffle_type'].item()})')
ax.legend()
g.plt.show()


# %%
# ---- Cell 3: plot test loss vs. index ----
from p_pack import globals as g
from pathlib import Path

test_loss_folder = "p2-reup-vary/test-acc-500"
file_indent = 't'
indexes_to_plot = [0, 1, 2, 3, 4,5,6,7,8,9]
legend_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

fig, ax = g.plt.subplots()

script_dir = g.os.getcwd()
parent_dir = g.os.path.abspath(g.os.path.join(script_dir, '..'))
test_folder_path = g.os.path.join(parent_dir, 'work', test_loss_folder)

test_vals = []
for idx, label in zip(indexes_to_plot, legend_names):
    matches = list(Path(test_folder_path).glob(f't{idx}*.npz'))
    if not matches:
        print(f'[Warning] Test loss file for index {idx} not found.')
        continue
    tl = g.np.load(matches[0])['test_loss'].item()
    test_vals.append(tl)
    ax.plot(idx, tl, 'o', label=label)

ax.set(xlabel='index', ylabel='test loss', title=f'Test Loss vs reup-f-2p(shuffle={gl['shuffle_type'].item()})')
ax.legend()
g.plt.show()


# %%
from p_pack import globals as g


# Mapping of photon number to their respective result directories
PHOTON_DIRS = {
    1: "p1-shuffle-rf4-800",
    2: "p2-shuffle-rf4-800",
    3: "p3-shuffle-rf4-800",
}

# Training runs that were evaluated
RUN_NUMBERS = [25, 50, 100, 200, 400, 800]


def _load_metric(photon_dir: str, metric: str, shuffle_type: int):
    """Load metric values for a photon configuration and shuffle type.

    Parameters
    ----------
    photon_dir:
        Root directory for the photon configuration (e.g. ``p1-shuffle-rf4-800``).
    metric:
        Either ``"test-loss"`` or ``"test-acc"``.
    shuffle_type:
        The shuffle index (0, 1 or 2).

    Returns
    -------
    tuple[list[int], list[float]]
        Lists of run numbers and the corresponding metric values.  Missing
        files are skipped with a warning.
    """

    # Build absolute path to the photon directory within the external ``work`` folder
    script_dir = g.os.getcwd()
    parent_dir = g.os.path.abspath(g.os.path.join(script_dir, ".."))
    base_path = Path(parent_dir) / "work" / photon_dir

    runs, values = [], []
    for run in RUN_NUMBERS:
        file_path = base_path / f"{metric}-{run}" / f"t{shuffle_type}s{shuffle_type}.npz"
        if not file_path.exists():
            print(f"[Warning] Missing file: {file_path}")
            continue
        with g.np.load(file_path, allow_pickle=True) as data:
            # Files saved by ``evaluate_and_save_test_loss`` always use the
            # ``test_loss`` key even when storing accuracy.
            values.append(data["test_loss"].item())
            runs.append(run)
    return runs, values


def plot_results():
    """Create figures for each shuffle type showing loss and accuracy."""

    labels = {1: "1 photon", 2: "2 photons", 3: "3 photons"}

    for shuffle in [0, 1, 2]:
        fig, (ax_loss, ax_acc) = g.plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        for p, directory in PHOTON_DIRS.items():
            run_loss, loss_vals = _load_metric(directory, "test-loss", shuffle)
            run_acc, acc_vals = _load_metric(directory, "test-acc", shuffle)

            ax_loss.plot(run_loss, loss_vals, marker="o", label=labels[p])
            ax_acc.plot(run_acc, acc_vals, marker="o", label=labels[p])

        ax_loss.set_ylabel("Test Loss")
        ax_loss.legend()
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_xlabel("Run Number")
        fig.suptitle(f"Shuffle {shuffle}")
        g.plt.tight_layout()
        g.plt.show()

plot_results()

# %%
from pathlib import Path
from p_pack import globals as g


# Mapping of photon number to their respective result directories
PHOTON_DIRS = {
    1: "p1-shuffle-rf3-800",
    2: "p2-shuffle-rf3-800",
    3: "p3-shuffle-rf3-800",
}

# Explicit colors so photon and shuffle plots do not share palettes
PHOTON_COLORS = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
SHUFFLE_COLORS = {0: "tab:red", 1: "tab:purple", 2: "tab:brown"}

# Line styles for photon comparison plots (all dotted with increasing frequency)
PHOTON_LINESTYLES = {
    1: (0, (5, 5)),  # least frequent dots
    2: (0, (3, 3)),  # medium frequency
    3: (0, (1, 1)),  # most frequent dots
}

# Training runs that were evaluated
RUN_NUMBERS = [25, 50, 100, 200, 400, 800]


def _load_metric(photon_dir: str, metric: str, shuffle_type: int):
    """Load metric values for a photon configuration and shuffle type.

    Parameters
    ----------
    photon_dir:
        Root directory for the photon configuration (e.g. ``p1-shuffle-rf4-800``).
    metric:
        Either ``"test-loss"`` or ``"test-acc"``.
    shuffle_type:
        The shuffle index (0, 1 or 2).

    Returns
    -------
    tuple[list[int], list[float]]
        Lists of run numbers and the corresponding metric values.  Missing
        files are skipped with a warning.
    """

    # Build absolute path to the photon directory within the external ``work`` folder
    script_dir = g.os.getcwd()
    parent_dir = g.os.path.abspath(g.os.path.join(script_dir, ".."))
    base_path = Path(parent_dir) / "work" / photon_dir

    runs, values = [], []
    for run in RUN_NUMBERS:
        file_path = (
            base_path / f"{metric}-{run}" / f"t{shuffle_type}s{shuffle_type}.npz"
        )
        if not file_path.exists():
            print(f"[Warning] Missing file: {file_path}")
            continue
        with g.np.load(file_path, allow_pickle=True) as data:
            # Files saved by ``evaluate_and_save_test_loss`` always use the
            # ``test_loss`` key even when storing accuracy.
            values.append(data["test_loss"].item())
            runs.append(run)
    return runs, values


def _plot_multicolor(ax, x, y, marker="o"):
    """Plot a line with a rainbow colormap along its length."""

    if len(x) < 2:
        ax.plot(x, y, marker=marker)
        return

    colors = g.plt.cm.rainbow(g.np.linspace(0, 1, len(x) - 1))
    for i in range(len(x) - 1):
        ax.plot(x[i : i + 2], y[i : i + 2], color=colors[i], marker=marker)


def plot_results():
    """Create figures for each shuffle type showing loss and accuracy."""

    labels = {1: "1 photon", 2: "2 photon", 3: "3 photons"}

    for shuffle in [0, 1, 2]:
        fig, (ax_loss, ax_acc) = g.plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        for p, directory in PHOTON_DIRS.items():
            run_loss, loss_vals = _load_metric(directory, "test-loss", shuffle)
            run_acc, acc_vals = _load_metric(directory, "test-acc", shuffle)

            ax_loss.plot(
                run_loss,
                loss_vals,
                marker="o",
                label=labels[p],
                color=PHOTON_COLORS[p],
                linestyle=PHOTON_LINESTYLES[p],
            )
            ax_acc.plot(
                run_acc,
                acc_vals,
                marker="o",
                label=labels[p],
                color=PHOTON_COLORS[p],
                linestyle=PHOTON_LINESTYLES[p],
            )

        ax_loss.set_ylabel("Test Loss")
        ax_loss.legend()
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_xlabel("Run Number")
        fig.suptitle(f"Shuffle {shuffle}")
        g.plt.tight_layout()
        g.plt.show()


def plot_photon_results():
    """Create figures for each photon number showing loss and accuracy."""

    labels = {0: "shuffle 0", 1: "shuffle 1", 2: "shuffle 2"}

    for p, directory in PHOTON_DIRS.items():
        fig, (ax_loss, ax_acc) = g.plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        for shuffle in [0, 1, 2]:
            run_loss, loss_vals = _load_metric(directory, "test-loss", shuffle)
            run_acc, acc_vals = _load_metric(directory, "test-acc", shuffle)

            if shuffle == 0:
                _plot_multicolor(ax_loss, run_loss, loss_vals)
                _plot_multicolor(ax_acc, run_acc, acc_vals)
            else:
                linestyle = ":" if shuffle == 2 else "-"
                ax_loss.plot(
                    run_loss,
                    loss_vals,
                    marker="o",
                    label=labels[shuffle],
                    color=SHUFFLE_COLORS[shuffle],
                    linestyle=linestyle,
                )
                ax_acc.plot(
                    run_acc,
                    acc_vals,
                    marker="o",
                    label=labels[shuffle],
                    color=SHUFFLE_COLORS[shuffle],
                    linestyle=linestyle,
                )

        handles = [
            g.plt.Line2D([], [], color=SHUFFLE_COLORS[0], label=labels[0]),
            g.plt.Line2D(
                [], [], color=SHUFFLE_COLORS[1], linestyle="-", label=labels[1]
            ),
            g.plt.Line2D(
                [], [], color=SHUFFLE_COLORS[2], linestyle=":", label=labels[2]
            ),
        ]
        ax_loss.set_ylabel("Test Loss")
        ax_loss.legend(handles=handles)
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_xlabel("Run Number")
        title = f"{p} photon" if p == 1 else f"{p} photons"
        fig.suptitle(title)
        g.plt.tight_layout()
        g.plt.show()

plot_results()
plot_photon_results()


# %%

from p_pack import globals as g
from pathlib import Path

# Mapping of photon number to their respective result directories
PHOTON_DIRS = {
    1: "p1-shuffle-rf8-800-d10",
    2: "p2-shuffle-rf8-800-d10",
    3: "p3-shuffle-rf8-800-d10",
}

# Training runs that were evaluated
RUN_NUMBERS = [25, 50, 100, 200, 400, 800]


def _load_metric(photon_dir: str, metric: str, shuffle_type: int):
    """Load metric values for a photon configuration and shuffle type.

    Parameters
    ----------
    photon_dir:
        Root directory for the photon configuration (e.g. ``p1-shuffle-rf4-800``).
    metric:
        Either ``"test-loss"`` or ``"test-acc"``.
    shuffle_type:
        The shuffle index (0, 1 or 2).

    Returns
    -------
    tuple[list[int], list[float]]
        Lists of run numbers and the corresponding metric values.  Missing
        files are skipped with a warning.
    """

    # Build absolute path to the photon directory within the external ``work`` folder
    script_dir = g.os.getcwd()
    parent_dir = g.os.path.abspath(g.os.path.join(script_dir, ".."))
    base_path = Path(parent_dir) / "work" / photon_dir

    runs, values = [], []
    for run in RUN_NUMBERS:
        file_path = base_path / f"{metric}-{run}" / f"t{shuffle_type}s{shuffle_type}.npz"
        if not file_path.exists():
            print(f"[Warning] Missing file: {file_path}")
            continue
        with g.np.load(file_path, allow_pickle=True) as data:
            # Files saved by ``evaluate_and_save_test_loss`` always use the
            # ``test_loss`` key even when storing accuracy.
            values.append(data["test_loss"].item())
            runs.append(run)
    return runs, values


def plot_results():
    """Create figures for each shuffle type showing loss and accuracy."""

    labels = {1: "1 photon", 2: "2 photons", 3: "3 photons"}

    for shuffle in [0, 1, 2]:
        fig, (ax_loss, ax_acc) = g.plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        for p, directory in PHOTON_DIRS.items():
            run_loss, loss_vals = _load_metric(directory, "test-loss", shuffle)
            run_acc, acc_vals = _load_metric(directory, "test-acc", shuffle)

            ax_loss.plot(run_loss, loss_vals, marker="o", label=labels[p])
            ax_acc.plot(run_acc, acc_vals, marker="o", label=labels[p])

        ax_loss.set_ylabel("Test Loss")
        ax_loss.legend()
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_xlabel("Run Number")
        fig.suptitle(f"Shuffle {shuffle}")
        g.plt.tight_layout()
        g.plt.show()

plot_results()

# %%


# %%


# %%
# ---- Cell 2: init_phases-2 curves with optional test-loss lines ----
from p_pack import globals as g
from pathlib import Path

# -------- Configuration --------
data_folders = ("p1-shuffle-rf8-800-d10/Learning","p2-shuffle-rf8-800-d10/Learning","p3-shuffle-rf8-800-d10/Learning")
for i in data_folders:
    data_folder = i
    file_indent = 's'
    indexes_to_plot = [0,1,2]  # Choose the indexes you want to plot
    legend_names = ["0","1","2"]  # Custom labels, must match index order
    x_min = 280
    x_max = 300
    y_min = 0.05
    y_max = 0.15
    plot_test_loss = 0 # set to 1 to show test loss lines
    test_loss_folder = "p1-position0-vs-shuffle-rf2/test-pos0"

    # Create two side-by-side plots
    fig, (ax_full, ax_zoom) = g.plt.subplots(1, 2, figsize=(12, 5))

    # Set data folder path
    script_dir = g.os.getcwd()
    parent_dir = g.os.path.abspath(g.os.path.join(script_dir, ".."))
    data_folder_path = g.os.path.join(parent_dir, "work", data_folder)
    test_folder_path = g.os.path.join(parent_dir, "work", test_loss_folder)

    # Loop over the selected indexes and legend labels
    for idx, label in zip(indexes_to_plot, legend_names):
        matches = list(Path(data_folder_path).glob(f"it{idx}{file_indent}*g.npz"))
        if not matches:
            print(f"[Warning] Global file for index {idx} not found.")
            continue

        global_name = matches[0].name
        data_name   = global_name.replace("g.npz", ".npz")

        data_path    = g.os.path.join(data_folder_path, data_name)
        globals_path = g.os.path.join(data_folder_path, global_name)

        out    = g.np.load(data_path)
        steps  = out["loss_mem"][:, 0].astype(int)
        losses = out["loss_mem"][:, 1].astype(float)

        # plot training loss and capture its color
        line_full, = ax_full.plot(steps, losses, linewidth=1.5, label=label)
        line_zoom, = ax_zoom.plot(steps, losses, linewidth=1.5, label=label)
        color = line_full.get_color()

        if plot_test_loss:
            test_name = 't' + global_name[2:].replace('g.npz', '.npz')
            test_path = g.os.path.join(test_folder_path, test_name)
            if g.os.path.exists(test_path):
                tl = g.np.load(test_path)['test_loss'].item()
                # draw test-loss as a horizontal line in the same color
                ax_full.axline((0, tl), (1, tl),
                            linestyle='--',
                            color=color,
                            label=f'test {label}')
                ax_zoom.axline((0, tl), (1, tl),
                            linestyle='--',
                            color=color)
            else:
                print(f"[Warning] Test loss file {test_name} not found.")

    # finalize axes
    ax_full.set(xlabel='step', ylabel='loss', title='Full Training Curve')
    ax_zoom.set(xlim=(x_min, x_max), ylim=(y_min, y_max),
                xlabel='step', ylabel='loss',
                title=f'Zoomed (steps {x_min}–{x_max})')

    handles, labels = ax_full.get_legend_handles_labels()
    ax_full.legend(handles, labels, loc='upper right', ncol=2, fontsize='small')

    gl    = g.np.load(globals_path)
    shape = out['carry_0'].shape
    fig.suptitle(
       # f"Unitary dim: {2*shape[1]}×{2*shape[1]}, p={gl['p_suc_inputs'].item()}, input=2 shuffle={gl['shuffle_type'].item()}, rf={gl['reupload_freq'].item()}"
       "10x10 shuffle 0, rf 4, varying photon number "
    )

    g.plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    g.plt.show()



# %%
from pathlib import Path
from p_pack import globals as g


# Mapping of photon number to their respective result directories
PHOTON_DIRS = {
    1: "p1-shuffle-rf8-800-d10",
    2: "p2-shuffle-rf9-800-d10",
    3: "p3-shuffle-rf9-800-d10",
}

# Explicit colors so photon and shuffle plots do not share palettes
PHOTON_COLORS = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
SHUFFLE_COLORS = {0: "tab:red", 1: "tab:purple", 2: "tab:brown"}

# Line styles for photon comparison plots (all dotted with increasing frequency)
PHOTON_LINESTYLES = {
    1: (0, (5, 5)),  # least frequent dots
    2: (0, (3, 3)),  # medium frequency
    3: (0, (1, 1)),  # most frequent dots
}

# Training runs that were evaluated
RUN_NUMBERS = [25, 50, 100, 200, 400, 800]


def _load_metric(photon_dir: str, metric: str, shuffle_type: int):
    """Load metric values for a photon configuration and shuffle type.

    Parameters
    ----------
    photon_dir:
        Root directory for the photon configuration (e.g. ``p1-shuffle-rf4-800``).
    metric:
        Either ``"test-loss"`` or ``"test-acc"``.
    shuffle_type:
        The shuffle index (0, 1 or 2).

    Returns
    -------
    tuple[list[int], list[float]]
        Lists of run numbers and the corresponding metric values.  Missing
        files are skipped with a warning.
    """

    # Build absolute path to the photon directory within the external ``work`` folder
    script_dir = g.os.getcwd()
    parent_dir = g.os.path.abspath(g.os.path.join(script_dir, ".."))
    base_path = Path(parent_dir) / "work" / photon_dir

    runs, values = [], []
    for run in RUN_NUMBERS:
        file_path = (
            base_path / f"{metric}-{run}" / f"t{shuffle_type}s{shuffle_type}.npz"
        )
        if not file_path.exists():
            print(f"[Warning] Missing file: {file_path}")
            continue
        with g.np.load(file_path, allow_pickle=True) as data:
            # Files saved by ``evaluate_and_save_test_loss`` always use the
            # ``test_loss`` key even when storing accuracy.
            values.append(data["test_loss"].item())
            runs.append(run)
    return runs, values


def _plot_multicolor(ax, x, y, marker="o"):
    """Plot a line with a rainbow colormap along its length."""

    if len(x) < 2:
        ax.plot(x, y, marker=marker)
        return

    colors = g.plt.cm.rainbow(g.np.linspace(0, 1, len(x) - 1))
    for i in range(len(x) - 1):
        ax.plot(x[i : i + 2], y[i : i + 2], color=colors[i], marker=marker)


def plot_results():
    """Create figures for each shuffle type showing loss and accuracy."""

    labels = {1: "1 photon", 2: "2 photon", 3: "3 photons"}

    for shuffle in [0, 1, 2]:
        fig, (ax_loss, ax_acc) = g.plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        for p, directory in PHOTON_DIRS.items():
            run_loss, loss_vals = _load_metric(directory, "test-loss", shuffle)
            run_acc, acc_vals = _load_metric(directory, "test-acc", shuffle)

            ax_loss.plot(
                run_loss,
                loss_vals,
                marker="o",
                label=labels[p],
                color=PHOTON_COLORS[p],
                linestyle=PHOTON_LINESTYLES[p],
            )
            ax_acc.plot(
                run_acc,
                acc_vals,
                marker="o",
                label=labels[p],
                color=PHOTON_COLORS[p],
                linestyle=PHOTON_LINESTYLES[p],
            )

        ax_loss.set_ylabel("Test Loss")
        ax_loss.legend()
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_xlabel("Run Number")
        fig.suptitle(f"Shuffle {shuffle}")
        g.plt.tight_layout()
        g.plt.show()


def plot_photon_results():
    """Create figures for each photon number showing loss and accuracy."""

    labels = {0: "shuffle 0", 1: "shuffle 1", 2: "shuffle 2"}

    for p, directory in PHOTON_DIRS.items():
        fig, (ax_loss, ax_acc) = g.plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        for shuffle in [0, 1, 2]:
            run_loss, loss_vals = _load_metric(directory, "test-loss", shuffle)
            run_acc, acc_vals = _load_metric(directory, "test-acc", shuffle)

            if shuffle == 0:
                _plot_multicolor(ax_loss, run_loss, loss_vals)
                _plot_multicolor(ax_acc, run_acc, acc_vals)
            else:
                linestyle = ":" if shuffle == 2 else "-"
                ax_loss.plot(
                    run_loss,
                    loss_vals,
                    marker="o",
                    label=labels[shuffle],
                    color=SHUFFLE_COLORS[shuffle],
                    linestyle=linestyle,
                )
                ax_acc.plot(
                    run_acc,
                    acc_vals,
                    marker="o",
                    label=labels[shuffle],
                    color=SHUFFLE_COLORS[shuffle],
                    linestyle=linestyle,
                )

        handles = [
            g.plt.Line2D([], [], color=SHUFFLE_COLORS[0], label=labels[0]),
            g.plt.Line2D(
                [], [], color=SHUFFLE_COLORS[1], linestyle="-", label=labels[1]
            ),
            g.plt.Line2D(
                [], [], color=SHUFFLE_COLORS[2], linestyle=":", label=labels[2]
            ),
        ]
        ax_loss.set_ylabel("Test Loss")
        ax_loss.legend(handles=handles)
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_xlabel("Run Number")
        title = f"{p} photon" if p == 1 else f"{p} photons"
        fig.suptitle(title)
        g.plt.tight_layout()
        g.plt.show()

plot_results()
plot_photon_results()


# %%
from pathlib import Path
from p_pack import globals as g


# Mapping of photon number to their respective result directories
PHOTON_DIRS = {
    1: "p1-shuffle-rf4-800",
    2: "reup-tuple-test",
    3: "p",
}

# Explicit colors so photon and shuffle plots do not share palettes
PHOTON_COLORS = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
SHUFFLE_COLORS = {0: "tab:red", 1: "tab:purple", 2: "tab:brown"}

# Line styles for photon comparison plots (all dotted with increasing frequency)
PHOTON_LINESTYLES = {
    1: (0, (5, 5)),  # least frequent dots
    2: (0, (3, 3)),  # medium frequency
    3: (0, (1, 1)),  # most frequent dots
}

# Training runs that were evaluated
RUN_NUMBERS = [25, 50, 100, 200, 400, 800]


def _load_metric(photon_dir: str, metric: str, shuffle_type: int):
    """Load metric values for a photon configuration and shuffle type.

    Parameters
    ----------
    photon_dir:
        Root directory for the photon configuration (e.g. ``p1-shuffle-rf4-800``).
    metric:
        Either ``"test-loss"`` or ``"test-acc"``.
    shuffle_type:
        The shuffle index (0, 1 or 2).

    Returns
    -------
    tuple[list[int], list[float]]
        Lists of run numbers and the corresponding metric values.  Missing
        files are skipped with a warning.
    """

    # Build absolute path to the photon directory within the external ``work`` folder
    script_dir = g.os.getcwd()
    parent_dir = g.os.path.abspath(g.os.path.join(script_dir, ".."))
    base_path = Path(parent_dir) / "work" / photon_dir

    runs, values = [], []
    for run in RUN_NUMBERS:
        file_path = (
            base_path / f"{metric}-{run}" / f"t{0}s{1}.npz"
        )
        if not file_path.exists():
            print(f"[Warning] Missing file: {file_path}")
            continue
        with g.np.load(file_path, allow_pickle=True) as data:
            # Files saved by ``evaluate_and_save_test_loss`` always use the
            # ``test_loss`` key even when storing accuracy.
            values.append(data["test_loss"].item())
            runs.append(run)
    return runs, values


def _plot_multicolor(ax, x, y, marker="o"):
    """Plot a line with a rainbow colormap along its length."""

    if len(x) < 2:
        ax.plot(x, y, marker=marker)
        return

    colors = g.plt.cm.rainbow(g.np.linspace(0, 1, len(x) - 1))
    for i in range(len(x) - 1):
        ax.plot(x[i : i + 2], y[i : i + 2], color=colors[i], marker=marker)


def plot_results():
    """Create figures for each shuffle type showing loss and accuracy."""

    labels = {1: "1 photon", 2: "2 photon", 3: "3 photons"}

    for shuffle in [0, 1, 2]:
        fig, (ax_loss, ax_acc) = g.plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        for p, directory in PHOTON_DIRS.items():
            run_loss, loss_vals = _load_metric(directory, "test-loss", shuffle)
            run_acc, acc_vals = _load_metric(directory, "test-acc", shuffle)

            ax_loss.plot(
                run_loss,
                loss_vals,
                marker="o",
                label=labels[p],
                color=PHOTON_COLORS[p],
                linestyle=PHOTON_LINESTYLES[p],
            )
            ax_acc.plot(
                run_acc,
                acc_vals,
                marker="o",
                label=labels[p],
                color=PHOTON_COLORS[p],
                linestyle=PHOTON_LINESTYLES[p],
            )

        ax_loss.set_ylabel("Test Loss")
        ax_loss.legend()
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_xlabel("Run Number")
        fig.suptitle(f"Shuffle {shuffle}")
        g.plt.tight_layout()
        g.plt.show()


def plot_photon_results():
    """Create figures for each photon number showing loss and accuracy."""

    labels = {0: "shuffle 0", 1: "shuffle 1", 2: "shuffle 2"}

    for p, directory in PHOTON_DIRS.items():
        fig, (ax_loss, ax_acc) = g.plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        for shuffle in [0,1,2]:
            run_loss, loss_vals = _load_metric(directory, "test-loss", shuffle)
            run_acc, acc_vals = _load_metric(directory, "test-acc", shuffle)

            if shuffle == 0:
                _plot_multicolor(ax_loss, run_loss, loss_vals)
                _plot_multicolor(ax_acc, run_acc, acc_vals)
            else:
                linestyle = ":" if shuffle == 2 else "-"
                ax_loss.plot(
                    run_loss,
                    loss_vals,
                    marker="o",
                    label=labels[shuffle],
                    color=SHUFFLE_COLORS[shuffle],
                    linestyle=linestyle,
                )
                ax_acc.plot(
                    run_acc,
                    acc_vals,
                    marker="o",
                    label=labels[shuffle],
                    color=SHUFFLE_COLORS[shuffle],
                    linestyle=linestyle,
                )

        handles = [
            g.plt.Line2D([], [], color=SHUFFLE_COLORS[0], label=labels[0]),
            g.plt.Line2D(
                [], [], color=SHUFFLE_COLORS[1], linestyle="-", label=labels[1]
            ),
            g.plt.Line2D(
                [], [], color=SHUFFLE_COLORS[2], linestyle=":", label=labels[2]
            ),
        ]
        ax_loss.set_ylabel("Test Loss")
        ax_loss.legend(handles=handles)
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_xlabel("Run Number")
        title = f"{p} photon" if p == 1 else f"{p} photons"
        fig.suptitle(title)
        g.plt.tight_layout()
        g.plt.show()

plot_results()
plot_photon_results()


# %%
# ---- Cell 2: init_phases-2 curves with optional test-loss lines ----
from p_pack import globals as g
from pathlib import Path

# -------- Configuration --------
data_folders = ("d10-p3-long-test/Learning",)
for i in data_folders:
    data_folder = i
    file_indent = 's'
    indexes_to_plot = [1,2,3]  # Choose the indexes you want to plot
    legend_names = ["1", "2", "3"]  # Custom labels, must match index order
    x_min = 400
    x_max = 600
    y_min = 0.05
    y_max = 0.1
    plot_test_loss = 0 # set to 1 to show test loss lines
    test_loss_folder = "p1-position0-vs-shuffle-rf2/test-pos0"

    # Create two side-by-side plots
    fig, (ax_full, ax_zoom) = g.plt.subplots(1, 2, figsize=(12, 5))

    # Set data folder path
    script_dir = g.os.getcwd()
    parent_dir = g.os.path.abspath(g.os.path.join(script_dir, ".."))
    data_folder_path = g.os.path.join(parent_dir, "work", data_folder)
    test_folder_path = g.os.path.join(parent_dir, "work", test_loss_folder)

    # Loop over the selected indexes and legend labels
    for idx, label in zip(indexes_to_plot, legend_names):
        matches = list(Path(data_folder_path).glob(f"it0s1g.npz"))
        if not matches:
            print(f"[Warning] Global file for index {idx} not found.")
            continue

        global_name = matches[0].name
        data_name   = global_name.replace("g.npz", ".npz")

        data_path    = g.os.path.join(data_folder_path, data_name)
        globals_path = g.os.path.join(data_folder_path, global_name)

        out    = g.np.load(data_path)
        steps  = out["loss_mem"][:, 0].astype(int)
        losses = out["loss_mem"][:, 1].astype(float)

        # plot training loss and capture its color
        line_full, = ax_full.plot(steps, losses, linewidth=1.5, label=label)
        line_zoom, = ax_zoom.plot(steps, losses, linewidth=1.5, label=label)
        color = line_full.get_color()

        if plot_test_loss:
            test_name = 't' + global_name[2:].replace('g.npz', '.npz')
            test_path = g.os.path.join(test_folder_path, test_name)
            if g.os.path.exists(test_path):
                tl = g.np.load(test_path)['test_loss'].item()
                # draw test-loss as a horizontal line in the same color
                ax_full.axline((0, tl), (1, tl),
                            linestyle='--',
                            color=color,
                            label=f'test {label}')
                ax_zoom.axline((0, tl), (1, tl),
                            linestyle='--',
                            color=color)
            else:
                print(f"[Warning] Test loss file {test_name} not found.")

    # finalize axes
    ax_full.set(xlabel='step', ylabel='loss', title='Full Training Curve')
    ax_zoom.set(xlim=(x_min, x_max), ylim=(y_min, y_max),
                xlabel='step', ylabel='loss',
                title=f'Zoomed (steps {x_min}–{x_max})')

    handles, labels = ax_full.get_legend_handles_labels()
    ax_full.legend(handles, labels, loc='upper right', ncol=2, fontsize='small')

    gl    = g.np.load(globals_path)
    shape = out['carry_0'].shape
    fig.suptitle(
       # f"Unitary dim: {2*shape[1]}×{2*shape[1]}, p={gl['p_suc_inputs'].item()}, input=2 shuffle={gl['shuffle_type'].item()}, rf={gl['reupload_freq'].item()}"
       "10x10 shuffle 0, rf 4, varying photon number "
    )

    g.plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    g.plt.show()



# %%
# ---- Cell 2: init_phases-2 curves with optional test-loss lines ----
from p_pack import globals as g
from pathlib import Path

# -------- Configuration --------
data_folders = ("sample-pos-test/Learning",)
for i in data_folders:
    data_folder = i
    file_indent = 's'
    indexes_to_plot = [0,1]  # Choose the indexes you want to plot
    legend_names = ["1", "2", "3"]  # Custom labels, must match index order
    x_min = 400
    x_max = 600
    y_min = 0.05
    y_max = 0.1
    plot_test_loss = 0 # set to 1 to show test loss lines
    test_loss_folder = "p1-position0-vs-shuffle-rf2/test-pos0"

    # Create two side-by-side plots
    fig, (ax_full, ax_zoom) = g.plt.subplots(1, 2, figsize=(12, 5))

    # Set data folder path
    script_dir = g.os.getcwd()
    parent_dir = g.os.path.abspath(g.os.path.join(script_dir, ".."))
    data_folder_path = g.os.path.join(parent_dir, "work", data_folder)
    test_folder_path = g.os.path.join(parent_dir, "work", test_loss_folder)

    # Loop over the selected indexes and legend labels
    for idx, label in zip(indexes_to_plot, legend_names):
        matches = list(Path(data_folder_path).glob(f"it{idx}{file_indent}*g.npz"))
        if not matches:
            print(f"[Warning] Global file for index {idx} not found.")
            continue

        global_name = matches[0].name
        data_name   = global_name.replace("g.npz", ".npz")

        data_path    = g.os.path.join(data_folder_path, data_name)
        globals_path = g.os.path.join(data_folder_path, global_name)

        out    = g.np.load(data_path)
        steps  = out["loss_mem"][:, 0].astype(int)
        losses = out["loss_mem"][:, 1].astype(float)

        # plot training loss and capture its color
        line_full, = ax_full.plot(steps, losses, linewidth=1.5, label=label)
        line_zoom, = ax_zoom.plot(steps, losses, linewidth=1.5, label=label)
        color = line_full.get_color()

        if plot_test_loss:
            test_name = 't' + global_name[2:].replace('g.npz', '.npz')
            test_path = g.os.path.join(test_folder_path, test_name)
            if g.os.path.exists(test_path):
                tl = g.np.load(test_path)['test_loss'].item()
                # draw test-loss as a horizontal line in the same color
                ax_full.axline((0, tl), (1, tl),
                            linestyle='--',
                            color=color,
                            label=f'test {label}')
                ax_zoom.axline((0, tl), (1, tl),
                            linestyle='--',
                            color=color)
            else:
                print(f"[Warning] Test loss file {test_name} not found.")

    # finalize axes
    ax_full.set(xlabel='step', ylabel='loss', title='Full Training Curve')
    ax_zoom.set(xlim=(x_min, x_max), ylim=(y_min, y_max),
                xlabel='step', ylabel='loss',
                title=f'Zoomed (steps {x_min}–{x_max})')

    handles, labels = ax_full.get_legend_handles_labels()
    ax_full.legend(handles, labels, loc='upper right', ncol=2, fontsize='small')

    gl    = g.np.load(globals_path)
    shape = out['carry_0'].shape
    fig.suptitle(
       # f"Unitary dim: {2*shape[1]}×{2*shape[1]}, p={gl['p_suc_inputs'].item()}, input=2 shuffle={gl['shuffle_type'].item()}, rf={gl['reupload_freq'].item()}"
       "10x10 shuffle 0, rf 4, varying photon number "
    )

    g.plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    g.plt.show()




