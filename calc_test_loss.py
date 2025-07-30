from pathlib import Path
from p_pack import globals as g
from p_pack import utils

# ----- Evaluation configuration -----
output_folder_name = "symmetry-true-2p"
results_subfolder = "test-p2"

# custom input configuration used for evaluation
input_positions = [0,9]
num_modes_circ = 10
p_suc_inputs = 1
input_config = g.input_config_maker(input_positions, num_modes_circ, p_suc_inputs)

# Location of training outputs
output_folder = str(Path.home() / "work" / output_folder_name)


def iterate_models(folder: str, subfolder: str, inp_conf):
    params_dir = Path(folder) / "ModelParams"
    learning_dir = Path(folder) / "Learning"
    for param_file in sorted(params_dir.glob("*.npz")):
        base = param_file.name
        core = base[1:] if base.startswith("m") else base
        cand1 = learning_dir / ("it" + core.replace(".npz", "g.npz"))
        cand2 = learning_dir / (core.replace(".npz", "g.npz"))
        globals_path = None
        for c in (cand1, cand2):
            if c.exists():
                globals_path = c
                break
        if globals_path is None:
            print(f"No globals found for {param_file.name}, skipping")
            continue
        out_name = "t" + core
        out_path = Path(folder) / subfolder / out_name
        utils.evaluate_and_save_test_loss(
            str(param_file),
            str(globals_path),
            inp_conf,
            str(out_path),
        )
        print(base, "done")


if __name__ == "__main__":
    iterate_models(output_folder, results_subfolder, input_config)