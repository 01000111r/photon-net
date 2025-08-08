from pathlib import Path
from p_pack import globals as g
from p_pack import utils

# ----- Evaluation configuration -----
output_folder_names = ("p3-pos-sample-s-all",)
model_numbers = [1000]
hard_predict = True
input_positions = [0, 4, 9]
num_modes_circ = 10
p_suc_inputs = 1
input_config = g.input_config_maker(input_positions, num_modes_circ, p_suc_inputs)
average_input_combinations = True

if hard_predict:
    if average_input_combinations:
        test_name = "acc-av"
    else:
        test_name = "acc -std"
else:
    if average_input_combinations:
        test_name = "loss-av"
    else:
        test_name = "loss-std"


def iterate_models(folder: Path, subfolder: str, model_number: int, inp_conf):
    params_dir = Path(folder) / f"ModelParams{model_number}"
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
        out_path = folder / subfolder / out_name
        utils.evaluate_and_save_test_loss(
            str(param_file),
            str(globals_path),
            inp_conf,
            str(out_path),
            hard_predict=hard_predict,
        )
        print(base, "done")


if __name__ == "__main__":
    for output_folder_name in output_folder_names:
        output_folder = Path.home() / "work" / output_folder_name
        print(f"Processing folder: {output_folder}")
        for model_number in model_numbers:
            results_subfolder = f"test-{test_name}-{model_number}"
            full_results_path = output_folder / results_subfolder
            full_results_path.mkdir(parents=True, exist_ok=True)
            iterate_models(output_folder, results_subfolder, model_number, input_config)

