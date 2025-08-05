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
