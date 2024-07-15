import ast
import os
from pathlib import Path as P
from pathlib import PurePath as PP
from dataclasses import dataclass
from readDataFile import read

import PIL
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import pandas as pd
import scipy

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


def atom_numbers(filename):
    numbers = np.loadtxt(filename)
    return numbers.astype(int)


def load_locations(filename):
    locations = np.load(filename)
    return locations


def plot_atom(numbers, locations):
    ax = plt.figure().add_subplot(projection="3d")
    first = len(numbers) // 4  # middle of first residue
    second = len(numbers) // 4 * 3  # middle of second residue
    ax.plot3D(
        locations[:, first, 0],
        locations[:, first, 1],
        locations[:, first, 2],
        label="406",
    )  # Plot contour curves
    ax.plot3D(
        locations[:, second, 0],
        locations[:, second, 1],
        locations[:, second, 2],
        label="537",
    )  # Plot contour curves
    ax.legend()
    ax.set_ylabel("Y position (nm)")
    ax.set_xlabel("X position (nm)")
    ax.set_zlabel("Z position (nm)")
    fig2, ax2 = plt.subplots()
    diff = locations[:, first, :] - locations[:, second, :]
    dist = np.linalg.norm(diff, axis=1)

    t = np.linspace(200, 450, diff.shape[0])
    ax2.plot(t, dist)
    ax2.set_ylabel("406-537 distance (nm)")
    ax2.set_xlabel("Time (ns)")
    highspin = 2

    def coupling_strength(r):
        mu0 = 1.257e-6
        muB = 9.27e-24
        h = 6.63e-34
        g = 2
        pi = np.pi
        # use h not hbar because we want frequency not angular frequency units

        return highspin * mu0 / (4 * pi) * muB**2 * g**2 / h * 1 / r**3

    omega = coupling_strength(np.linalg.norm(diff, axis=1) * 1e-9)
    Edd = omega * (1 - 3 * (diff[:, 2] / np.linalg.norm(diff, axis=1)) ** 2)

    ang = np.arccos(diff[:, 2] / np.linalg.norm(diff, axis=1))  # - np.pi / 2

    fig3, ax3 = plt.subplots()
    ax3.set_ylabel("$E_{dd}$ (MHz)")
    ax3.set_xlabel("Time (ns)")
    ax3.plot(t, np.zeros(len(t)), ls="--", alpha=0.25, c="k")
    ax3.plot(
        t, np.ones(len(t)) * np.mean(omega), label=r"$\omega_{dd}$", ls="--"
    )
    ax3.plot(
        t,
        Edd,
        # / np.max(np.abs(Edd)),
        label="Coupling",
    )

    cum = [
        ii / (ind + 1)
        for ind, ii in enumerate(scipy.integrate.cumulative_trapezoid(Edd))
    ]
    ax3.plot(
        t[1:],
        cum,  # / np.max(np.abs((cum))),
        label="Cumulative",
    )

    ax4 = ax3.twinx()
    ax4.set_ylabel("Angle (rad)")
    ax4.plot(t, ang, label="Angle", c="k")
    fig3.legend(ncols=4)


def main():
    numbers = atom_numbers(
        "/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Code/dipolar averaging/MD simulations from Jackson/406-537_atom_numbers.txt"
    )
    locations = load_locations(
        "/Users/Brad/Library/CloudStorage/GoogleDrive-bdprice@ucsb.edu/My Drive/Research/Code/dipolar averaging/MD simulations from Jackson/406-537_locations.npy"
    )
    plot_atom(numbers, locations)


if __name__ == "__main__":
    main()
    plt.show()
