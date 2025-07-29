import os
import numpy as np
from jax import block_until_ready
from p_pack import train               # your train.train
import p_pack.globals as g

def save_run(log_file: str, output_folder: str, data_name: str, global_name, init_carry):
    """
    1) Runs train.train(init_carry)
    2) Saves the four outputs into output_folder/<data_name>.npz
    3) Saves the following globals (hard‑coded) into globals.npz:
         depth, num_features, p_suc_inputs, input_config, master_key,
         loss_function, aim, training_rate
    4) Appends a human‑readable entry to log_file
    """
    # 1) run the training
    carry, loss_mem, update_mem, n_p = block_until_ready(
        train.train(init_carry)
    )

    # 2) ensure output folder exists with subfolders
    runs_folder = os.path.join(output_folder, "Learning")
    params_folder = os.path.join(output_folder, "ModelParams")
    os.makedirs(runs_folder, exist_ok=True)
    os.makedirs(params_folder, exist_ok=True)

    # 3) save the four outputs into a .npz
    fname = data_name if data_name.endswith(".npz") else data_name + ".npz"
    outputs_path = os.path.join(runs_folder, "it" + fname)

    outputs_dict = {f"carry_{i}": np.asarray(x)
                    for i, x in enumerate(carry)}
    outputs_dict["loss_mem"]   = np.asarray(loss_mem)
    outputs_dict["update_mem"] = np.asarray(update_mem)
    outputs_dict["n_p"]        = np.asarray(n_p)

    np.savez_compressed(outputs_path, **outputs_dict)
    print(f"[save_run] Saved outputs → {outputs_path}")

    # 4) hard‑coded snapshot of key globals
    globals_path = os.path.join(runs_folder, "it" + global_name)
    to_save = {
        "num_steps":      np.asarray(g.num_steps),
        "training_rate":  np.asarray(g.training_rate),
        "reupload_freq":  np.asarray(g.reupload_freq),
        "num_modes_circ": np.asarray(g.num_modes_circ),
        "depth":          np.asarray(g.depth),
        "num_features":   np.asarray(g.num_features),
        "p_suc_inputs":   np.asarray(g.p_suc_inputs),
        "input_positions": np.asarray(g.input_positions),
        "input_config":   np.asarray(g.input_config[0]),
        "aim":            np.asarray(g.aim),
        "discard":        np.asarray(g.discard),
        "discard_condition": np.asarray(g.discard_condition),
        "discard_range":  np.asarray(g.discard_range),
        "loss_function": np.asarray(g.loss_function),
        "batch_mode":     np.asarray(g.batch_mode),
        "mini_batch_size": np.asarray(g.mini_batch_size),
        "master_key":     np.asarray(g.master_key),
        "phase_key":      np.asarray(g.phase_key),
        "phase_init_value":      np.asarray(g.phase_init_value)
    }
    np.savez_compressed(globals_path, **to_save)
    print(f"[save_run] Saved globals → {globals_path}")

    # Save final parameters (phases, weights, alpha)
    final_params = {
        "phases":  np.asarray(carry[0]),
        "weights": np.asarray(carry[3]),
        "alpha":   np.asarray(carry[4]),
    }
    params_path = os.path.join(params_folder, "m" + fname)
    np.savez_compressed(params_path, **final_params)
    print(f"[save_run] Saved final parameters → {params_path}")

    # 5) append a human‑readable entry to both the main and run logs
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    run_log_path = os.path.join(output_folder, "run_log.yaml")
    with open(log_file, "a") as f_main, open(run_log_path, "a") as f_out:
        for f in (f_main, f_out):
            f.write(f"\n=== Run: {os.path.basename(output_folder)} ===\n")
            f.write(f"Output folder: {output_folder}\n")
            f.write(f"Outputs file:  {outputs_path}\n")
            f.write(f"Globals file:  {globals_path}\n")
            f.write(f"Params file:   {params_path}\n\n")
            f.write("Globals snapshot:\n")
            for name, arr in to_save.items():
                if isinstance(arr, np.ndarray) and arr.size < 15:
                    summary = arr.tolist()
                else:
                    summary = f"<array shape={arr.shape}>"
                f.write(f"  {name}: {summary}\n")
            f.write("="*40 + "\n")
    print(f"[save_run] Appended entry to {log_file} and {run_log_path}")

    return carry, loss_mem, update_mem, n_p