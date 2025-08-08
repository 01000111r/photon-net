from __future__ import annotations

import itertools
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_test_runs(
    main_folder: str,
    subfolder_prefix: str,
    run_numbers: Iterable[int],
    shuffle_index: int,
):
    """Plot loss and accuracy for multiple test runs and summarise statistics.

    Parameters
    ----------
    main_folder:
        Name of the base experiment folder located under ``~/work``.
        Example: ``"s1-sample-pos-s-all"``.
    subfolder_prefix:
        Common prefix of the result folders.  For each run ``N`` the function
        expects to find folders ``{prefix}-acc-all-{N}`` and
        ``{prefix}-loss-all-{N}`` inside ``main_folder``.
    run_numbers:
        Iterable of run identifiers (e.g. ``[50, 100, 200]``) to overlay.
    shuffle_index:
        Index of the ``.npz`` file representing the shuffle type to plot.

    Returns
    -------
    fig : ``matplotlib.figure.Figure``
        Figure containing the overlaid loss and accuracy plots.
    loss_table, acc_table : ``pandas.DataFrame``
        Tables of mean±std for loss and accuracy.  Rows correspond to run
        numbers (sorted ascending) and columns to shuffle indices.
    """

    base = Path.home() / "work" / main_folder
    runs = sorted(run_numbers)

    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, sharex=True)

    loss_stats: dict[int, dict[int, Tuple[float, float]]] = {}
    acc_stats: dict[int, dict[int, Tuple[float, float]]] = {}
    shuffle_labels = None

    for run in runs:
        acc_folder = base / f"{subfolder_prefix}-acc-all-{run}"
        loss_folder = base / f"{subfolder_prefix}-loss-all-{run}"

        acc_files = sorted(acc_folder.glob("*.npz"))
        loss_files = sorted(loss_folder.glob("*.npz"))

        if shuffle_labels is None:
            shuffle_labels = list(range(len(acc_files)))

        if shuffle_index >= len(acc_files) or shuffle_index >= len(loss_files):
            raise IndexError(f"shuffle_index {shuffle_index} out of range for run {run}")

        with np.load(acc_files[shuffle_index]) as a_data, np.load(loss_files[shuffle_index]
        ) as l_data:
            acc_arr = a_data["accuracy_per_config"]
            loss_arr = l_data["loss_per_config"]
            x = np.arange(len(acc_arr))
            ax_acc.plot(x, acc_arr, marker="o", label=str(run))
            ax_loss.plot(x, loss_arr, marker="o", label=str(run))

        for idx, (a_file, l_file) in enumerate(zip(acc_files, loss_files)):
            with np.load(a_file) as a_data, np.load(l_file) as l_data:
                acc_mean = float(np.mean(a_data["accuracy_per_config"]))
                acc_std = float(np.std(a_data["accuracy_per_config"]))
                loss_mean = float(np.mean(l_data["loss_per_config"]))
                loss_std = float(np.std(l_data["loss_per_config"]))

            acc_stats.setdefault(run, {})[idx] = (acc_mean, acc_std)
            loss_stats.setdefault(run, {})[idx] = (loss_mean, loss_std)

    ax_loss.set_ylabel("Loss")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_xlabel("Combination index")
    ax_loss.set_title(f"Loss per combination (shuffle {shuffle_index})")
    ax_acc.set_title(f"Accuracy per combination (shuffle {shuffle_index})")
    ax_loss.legend(title="Run")
    ax_acc.legend(title="Run")
    fig.tight_layout()

    df_acc = pd.DataFrame(index=runs, columns=shuffle_labels)
    df_loss = pd.DataFrame(index=runs, columns=shuffle_labels)

    for run in runs:
        for idx in shuffle_labels:
            a_mean, a_std = acc_stats[run][idx]
            l_mean, l_std = loss_stats[run][idx]
            df_acc.loc[run, idx] = f"{a_mean:.3g}±{a_std:.3g}"
            df_loss.loc[run, idx] = f"{l_mean:.3g}±{l_std:.3g}"

    df_acc.index.name = "run"
    df_loss.index.name = "run"

    print("Accuracy mean ± std:")
    print(df_acc)
    print("\nLoss mean ± std:")
    print(df_loss)

    return fig, df_loss, df_acc



def combination_from_index(index: int, num_modes: int, photon_number: int):
    """Return the binary input combination for a given index.

    Parameters
    ----------
    index:
        0-based index of the combination.
    num_modes:
        Number of modes.
    photon_number:
        Number of photons in the configuration.
    """
    combos = list(itertools.combinations(range(num_modes), photon_number))
    if index < 0 or index >= len(combos):
        raise IndexError("index out of range")
    arr = []
    for idx in combos[index]:
        arr.append(idx)
    return tuple(arr)

main_folder = "p1-pos-sample-s-all"
subfolder_prefix = "test"
run_numbers = [50, 100, 200, 400, 800]
shuffle_index = 0

plot_test_runs(main_folder, subfolder_prefix, run_numbers, shuffle_index)

combination_from_index(0, 10, 2)