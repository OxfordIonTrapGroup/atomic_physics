"""Simple rate equations example of 393 shelving in 43Ca+."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

import atomic_physics as ap
from atomic_physics.ions import ca43


def main():
    t_ax = np.linspace(0, 100e-6, 100)
    I = 0.02  # 393 intensity

    ion = ca43.Ca43(B=146e-4)
    stretch = ion.index(ca43.ground_level, 4)

    rates = ap.rates.Rates(ion)
    delta = ion.delta(stretch, ion.index(ca43.P32, +5))
    lasers = [ap.Laser("393", q=+1, I=I, delta=delta)]  # resonant 393 sigma+
    trans = rates.get_transitions(lasers)

    Vi = np.zeros((ion.num_states, 1))  # initial state
    Vi[stretch] = 1  # start in F=4, M=+4
    shelved = np.zeros(len(t_ax))
    for idx, t in np.ndenumerate(t_ax):
        Vf = expm(trans * t) @ Vi
        shelved[idx] = sum(Vf[ion.slice(ca43.shelf)])

    plt.plot(t_ax * 1e6, shelved)
    plt.ylabel("Shelved Population")
    plt.xlabel("Shelving time (us)")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
