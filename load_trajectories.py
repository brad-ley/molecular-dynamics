import mdtraj as md
import numpy as np
from matplotlib import rc

if __name__ == "__main__":
    plt.style.use(["science"])
    rc("text.latex", preamble=r"\usepackage{cmbright}")
    rcParams = [
        ["font.family", "sans-serif"],
        ["font.size", 14],
        ["axes.linewidth", 1],
        ["lines.linewidth", 2],
        ["xtick.major.size", 5],
        ["xtick.major.width", 1],
        ["xtick.minor.size", 2],
        ["xtick.minor.width", 1],
        ["ytick.major.size", 5],
        ["ytick.major.width", 1],
        ["ytick.minor.size", 2],
        ["ytick.minor.width", 1],
    ]
    plt.rcParams.update(dict(rcParams))


def load(filename):
    t = md.load(
        "2V1A_Unfold_Solvate/sim-3kbar/full_trajec/npt_pbc_200-450ns.xtc",
        top="2V1A_Unfold_Solvate/sim-3kbar/full_trajec/npt_200-250ns.tpr",
    )

    print("MD Trajectory Shape (Num_Frames, Num_Atoms, Num_Coordinates):")
    print(t.xyz.shape)
    print("")
    print("Frame 0 (first 5 atoms)")
    print(t.xyz[0, :5, :])
    print("Frame 1 (first 5 atoms)")
    print(t.xyz[1, :5, :])
    print("")
    print("Frame -2 (first 5 atoms)")
    print(t.xyz[-2, :5, :])
    print("Frame -1 (first 5 atoms)")
    print(t.xyz[-1, :5, :])

    # Check radius of gyration
    rg_vals = md.compute_rg(t)
    print("Mean Radius of Gyration:", np.mean(rg_vals))  # TODO: Add mass array


def main(filename):
    return load(filename)


if __name__ == "__main__":
    filename = ""
    main(filename)
