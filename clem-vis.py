import argparse
from dataclasses import dataclass
from typing import List, Dict
import matplotlib.pyplot as plt

# Hardwired number of modes
NUM_MODES = 10

@dataclass
class BeamSplitter:
    mode1: int
    mode2: int

class Circuit:
    def __init__(self, num_layers: int):
        self.num_modes = NUM_MODES
        self.num_layers = num_layers
        self.layers: List[List[BeamSplitter]] = []
        self._build_clements_pattern()

    def _build_clements_pattern(self):
        # Clements pattern: alternating even/odd couplings
        for layer_idx in range(self.num_layers):
            bss: List[BeamSplitter] = []
            start = 0 if layer_idx % 2 == 0 else 1
            mode = start
            while mode + 1 < self.num_modes:
                bss.append(BeamSplitter(mode, mode + 1))
                mode += 2
            self.layers.append(bss)

    def get_beamsplitters(self, layer: int) -> List[BeamSplitter]:
        return self.layers[layer]

@dataclass
class PhotonPath:
    modes: List[int]


def compute_all_paths(
    num_layers: int,
    input_modes: List[int]
) -> Dict[int, List[PhotonPath]]:
    """
    Returns a mapping from each input mode to all possible PhotonPath instances
    through the 10-mode Clements circuit. Waveguides (straight paths) are only
    allowed at the extreme boundaries (mode 0 and mode NUM_MODES-1).
    """
    circuit = Circuit(num_layers)
    all_paths: Dict[int, List[PhotonPath]] = {}

    for inp in input_modes:
        partial_paths: List[List[int]] = [[inp]]

        for layer_idx in range(num_layers):
            next_paths: List[List[int]] = []
            bss = circuit.get_beamsplitters(layer_idx)
            for path in partial_paths:
                current_mode = path[-1]
                # Check for beamsplitter on this mode
                bs = next((bs for bs in bss if current_mode in (bs.mode1, bs.mode2)), None)
                if bs:
                    # Transmitted branch (stay)
                    next_paths.append(path + [current_mode])
                    # Reflected branch (cross)
                    other = bs.mode2 if current_mode == bs.mode1 else bs.mode1
                    next_paths.append(path + [other])
                else:
                    # Only allow straight (no coupling) if at the boundary modes
                    if current_mode in (0, NUM_MODES - 1):
                        next_paths.append(path + [current_mode])
                    # Internal unpaired modes are invalid and dropped
            partial_paths = next_paths

        all_paths[inp] = [PhotonPath(p) for p in partial_paths]

    return all_paths


def plot_paths(
    paths: Dict[int, List[PhotonPath]],
    num_layers: int
) -> None:
    """
    Draws each PhotonPath as a polyline over layer vs. mode indices.
    """
    plt.figure(figsize=(8, 6))
    # Horizontal guide lines for the 10 waveguides
    for m in range(NUM_MODES):
        plt.hlines(m, 0, num_layers, linestyles='dashed', linewidth=0.5)

    # Plot each path
    for inp, path_list in paths.items():
        for path_obj in path_list:
            xs = list(range(len(path_obj.modes)))
            ys = path_obj.modes
            plt.plot(xs, ys, alpha=0.7)

    plt.xlabel('Layer')
    plt.ylabel('Mode')
    plt.title(f'Photon Paths in 10-Mode Clements Circuit ({num_layers} layers)')
    plt.yticks(range(NUM_MODES))
    plt.xlim(0, num_layers)
    plt.ylim(-0.5, NUM_MODES - 0.5)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize photon paths in a 10-mode Clements interferometer.'
    )
    parser.add_argument(
        '--layers', '-l', type=int, required=True,
        help='Number of layers (stages of beam splitters)'
    )
    parser.add_argument(
        '--inputs', '-i', type=int, nargs='+', required=True,
        help='Input photon mode indices (0-9)'
    )

    args = parser.parse_args()
    paths = compute_all_paths(args.layers, args.inputs)
    plot_paths(paths, args.layers)


if __name__ == '__main__':
    main()
