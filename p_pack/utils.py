import os
from pathlib import Path
import numpy as np
from jax import block_until_ready
import jax.numpy as jnp
from p_pack import loss, model, train         # your train.train
import p_pack.globals as g
import itertools

def save_run(log_file: str, output_folder: str, data_name: str, global_name, init_carry):
    """
    1) Runs train.train(init_carry)
    2) Saves the four outputs into output_folder/<data_name>.npz
    3) Saves the following globals (hard‑coded) into globals.npz:
         depth, num_features, p_suc_inputs, input_config, master_key,
         loss_function, aim, training_rate
    4) Appends a human‑readable entry to log_file
    """
   # normalise list of save points and ensure final step is included
    total_steps = g.num_steps
    save_points = sorted(set(int(s) for s in g.save_points))
    if not save_points or save_points[-1] != total_steps:
        save_points.append(total_steps)

    carry = init_carry
    all_loss, all_update, all_n_p = [], [], []
    prev_step = 0

    fname = data_name if data_name.endswith(".npz") else data_name + ".npz"

    # run training in segments, saving parameters at requested steps
    for step in save_points:
        steps_to_run = step - prev_step
        if steps_to_run <= 0:
            continue

        carry, loss_mem, update_mem, n_p = block_until_ready(
            train.train(carry, num_steps=steps_to_run)
        )

        # adjust step numbers to account for previous segments
        # np.asarray may return a read-only view; copy so we can mutate
        loss_arr = np.asarray(loss_mem).copy()
        loss_arr[:, 0] += prev_step

        all_loss.append(loss_arr)
        all_update.append(np.asarray(update_mem))
        all_n_p.append(np.asarray(n_p))

        params_folder = os.path.join(output_folder, f"ModelParams{step}")
        os.makedirs(params_folder, exist_ok=True)

        params_path = os.path.join(params_folder, "m" + fname)
        final_params = {
            "phases": np.asarray(carry[0]),
            "weights": np.asarray(carry[3]),
            "alpha": np.asarray(carry[4]),
        }
        np.savez_compressed(params_path, **final_params)
        print(f"[save_run] Saved parameters at step {step} → {params_path}")

        # append log entry for this checkpoint
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        run_log_path = os.path.join(output_folder, "run_log.yaml")
        with open(log_file, "a") as f_main, open(run_log_path, "a") as f_out:
            for f in (f_main, f_out):
                f.write(f"\n=== Params at step {step} ===\n")
                f.write(f"Output folder: {output_folder}\n")
                f.write(f"Params file:   {params_path}\n")
                f.write("=" * 40 + "\n")

        prev_step = step

    # concatenate histories from all segments
    loss_mem = np.concatenate(all_loss) if all_loss else np.array([])
    update_mem = np.concatenate(all_update) if all_update else np.array([])
    n_p = np.concatenate(all_n_p) if all_n_p else np.array([])

    # save full learning history once
    runs_folder = os.path.join(output_folder, "Learning")
    os.makedirs(runs_folder, exist_ok=True)

    outputs_path = os.path.join(runs_folder, "it" + fname)
    outputs_dict = {f"carry_{i}": np.asarray(x) for i, x in enumerate(carry)}
    outputs_dict["loss_mem"] = loss_mem
    outputs_dict["update_mem"] = update_mem
    outputs_dict["n_p"] = n_p
    
    np.savez_compressed(outputs_path, **outputs_dict)
    print(f"[save_run] Saved outputs → {outputs_path}")

    # 4) hard‑coded snapshot of key globals
    globals_path = os.path.join(runs_folder, "it" + global_name)
    to_save = {
        "num_steps":      np.asarray(g.num_steps),
        "training_rate":  np.asarray(g.training_rate),
        "reupload_freq":  np.asarray(g.reupload_freq),
        "reup_is_tuple":  np.asarray(g.reup_is_tuple),
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
        "shuffle_key":    np.asarray(g.shuffle_key),
        "phase_init_value":      np.asarray(g.phase_init_value),
        "shuffle_type": np.asarray(g.shuffle_type),
        "symmetry_parity": np.asarray(g.use_symmetry_parity),
        "position_key": np.asarray(g.position_key),
        "position_sampling": np.asarray(g.position_sampling),
        "dataset_name": np.asarray(g.dataset_name),
        "class_labels": np.asarray(g.class_labels),
        "use_binary_labels": np.asarray(g.use_binary_labels),
        "num_classes": np.asarray(g.num_classes),
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

def evaluate_and_save_test_loss(
    params_path: str,
    globals_path: str,
    input_config,
    output_path: str,
    hard_predict: bool = False,
    average_input_combinations: bool = False,
    save_all_combinations: bool = False,
):
    """Compute test loss for a trained model and save it.

    Parameters
    ----------
    params_path : str
        Path to ``ModelParams`` ``.npz`` file containing ``phases``, ``weights``
        and ``alpha``.
    globals_path : str
        Path to ``globals`` ``.npz`` saved during training.
    input_config : tuple
        Input configuration to use for evaluation.
    output_path : str
        Location where the resulting ``test_loss`` will be stored.
    """

    # load globals and update runtime configuration
    if os.path.exists(globals_path):
        with np.load(globals_path, allow_pickle=True) as g_data:
            for name in g_data.files:
                if hasattr(g, name):
                    val = g_data[name]
                    if val.dtype == object and val.shape == ():
                        val = val.item()
                    setattr(g, name, val)
            # if hasattr(g, "master_key"):
            #     g.master_key = g.jnp.asarray(g.master_key)
            # if hasattr(g, "phase_key"):
            #     g.phase_key = g.jnp.asarray(g.phase_key)

    # some globals (like num_classes) depend on class_labels
    if hasattr(g, "class_labels"):
        g.num_classes = len(g.class_labels)

    # override the input configuration
    g.input_config = input_config

    params = np.load(params_path)
    phases = jnp.array(params["phases"])
    weights = jnp.array(params["weights"])
    alpha = float(params["alpha"])

    # load dataset based on the restored globals
    _, _, test_set, test_labels = g.final_load_data(g.num_features)

    test_set = jnp.array(test_set)
    test_labels = jnp.array(test_labels)

    if g.reup_is_tuple: reup_freq = tuple(g.reupload_freq) 
    else: reup_freq = int(g.reupload_freq)
    # reup_freq = int(g.reupload_freq)

    def _single_metrics(cfg):
        mask_local = jnp.asarray(cfg[0], dtype=jnp.int32)
        _, class_probs, n_p, _ = model.predict_reupload(
            phases,
            test_set,
            weights,
            cfg,
            mask_local,
            g.master_key,
            reup_freq,
            int(g.shuffle_type),
        )
        class_probs = jnp.asarray(class_probs)
        # predicted label indices
        pred_idx = jnp.argmax(class_probs, axis=1)
        class_array = jnp.asarray(g.class_labels)
        predicted_labels = class_array[pred_idx]
        correctness = (predicted_labels == test_labels).astype(jnp.float32)
        accuracy = jnp.mean(correctness) * 100.0

        labels_one_hot = (test_labels[:, None] == class_array[None, :]).astype(jnp.float32)
        loss_val = jnp.mean(jnp.sum((labels_one_hot - class_probs) ** 2, axis=1))
        return accuracy, loss_val

    if average_input_combinations:
        num_modes = len(input_config[0])
        photon_number = int(np.sum(input_config[0]))
        combos = itertools.combinations(range(num_modes), photon_number)
        accs, losses_list = [], []
        for combo in combos:
            arr = [0] * num_modes
            for idx in combo:
                arr[idx] = 1
            cfg = (tuple(arr), input_config[1])
            acc, l_val = _single_metrics(cfg)
            accs.append(acc)
            losses_list.append(l_val)
        if save_all_combinations:
            acc_arr = jnp.stack(accs) if accs else jnp.array([])
            loss_arr = jnp.stack(losses_list) if losses_list else jnp.array([])
            acc_val = jnp.mean(acc_arr) if accs else jnp.array(0.0)
            loss_val = jnp.mean(loss_arr) if losses_list else jnp.array(0.0)
        else:
            metrics = accs if hard_predict else losses_list
            loss_val = jnp.mean(jnp.stack(metrics)) if metrics else jnp.array(0.0)
            acc_val = loss_val if hard_predict else jnp.array(0.0)
    else:
        acc, l_val = _single_metrics(input_config)
        if save_all_combinations:
            acc_arr = jnp.array([acc])
            loss_arr = jnp.array([l_val])
            acc_val = acc
            loss_val = l_val
        else:
            loss_val = acc if hard_predict else l_val
            acc_val = acc if hard_predict else jnp.array(0.0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if save_all_combinations:
        np.savez_compressed(
            output_path,
            test_loss=np.asarray(loss_val),
            test_accuracy=np.asarray(acc_val),
            loss_per_config=np.asarray(loss_arr),
            accuracy_per_config=np.asarray(acc_arr),
        )
    else:
        np.savez_compressed(output_path, test_loss=np.asarray(loss_val))

    # prepare log file in the parent output folder
    root_folder = Path(output_path).resolve().parents[1]
    log_path = root_folder / "test_log.yaml"

    to_save = {
        "num_steps": np.asarray(g.num_steps),
        "training_rate": np.asarray(g.training_rate),
        "reupload_freq": np.asarray(g.reupload_freq),
        "reup_is_tuple": np.asarray(g.reup_is_tuple),
        "num_modes_circ": np.asarray(g.num_modes_circ),
        "depth": np.asarray(g.depth),
        "num_features": np.asarray(g.num_features),
        "p_suc_inputs": np.asarray(g.p_suc_inputs),
        "input_positions": np.asarray(g.input_positions),
        "input_config": np.asarray(g.input_config[0]),
        "aim": np.asarray(g.aim),
        "discard": np.asarray(g.discard),
        "discard_condition": np.asarray(g.discard_condition),
        "discard_range": np.asarray(g.discard_range),
        "loss_function": np.asarray(g.loss_function),
        "batch_mode": np.asarray(g.batch_mode),
        "mini_batch_size": np.asarray(g.mini_batch_size),
        "master_key": np.asarray(g.master_key),
        "phase_key": np.asarray(g.phase_key),
        "shuffle_key":    np.asarray(g.shuffle_key),
        "phase_init_value": np.asarray(g.phase_init_value),
        "shuffle_type": np.asarray(g.shuffle_type),
        "symmetry_parity": np.asarray(g.use_symmetry_parity),
        "accuracy": np.asarray(hard_predict),  # convert to percentage
        "average_input_combinations": np.asarray(average_input_combinations),
        "save_all_combinations": np.asarray(save_all_combinations),
    }

    with open(log_path, "a") as f:
        f.write(f"\n=== Test: {os.path.basename(params_path)} ===\n")
        f.write(f"Output folder: {root_folder}\n")
        f.write(f"Params file:   {params_path}\n")
        f.write(f"Globals file:  {globals_path}\n")
        f.write(f"Test loss file: {output_path}\n\n")
        f.write("Globals snapshot:\n")
        for name, arr in to_save.items():
            if isinstance(arr, np.ndarray) and arr.size < 15:
                summary = arr.tolist()
            else:
                summary = f"<array shape={arr.shape}>"
            f.write(f"  {name}: {summary}\n")
        f.write("=" * 40 + "\n")

    print(f"[evaluate_and_save_test_loss] Saved loss → {output_path}")
    return loss_val