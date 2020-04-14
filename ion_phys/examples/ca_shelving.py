""" Simple rate equations example of 393 shelving in 43Ca+. """
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from ion_phys.ions.ca43 import Ca43, ground_level, P32, shelf
from ion_phys import Laser
from ion_phys.rate_equations import Rates


def main():
    t_ax = np.linspace(0, 3e-6, 100)
    shelved = np.zeros(len(t_ax))
    shelved2 = np.zeros(len(t_ax))
    ion = Ca43(B=146e-4)
    stretch = ion.index(ground_level, 4)

    rates = Rates(ion)
    delta = ion.delta(stretch, ion.index(P32, +5))
    Lasers = [Laser("393", q=-1, I=100, delta=delta)]  # resonant 393 sigma+
    trans = rates.get_transitions(Lasers)
    # trans[np.abs(trans) < 1e-6] = 0

    for idx, t in np.ndenumerate(t_ax):
        Vi = np.zeros((ion.num_states, 1))  # initial state
        Vi[ion.index(P32, +5)] = 1  # start in F=4, M=+4
        Vf = expm(trans*t)@Vi
        # shelved[idx] = sum(Vf[ion.slice(shelf)])
        # print(trans[:, ion.index(P32, +5)])
        shelved[idx] = Vf[stretch]
        shelved2[idx] = Vf[ion.index(P32, +5)]
    Vf[np.abs(Vf) < 1e-3] = 0
    # print(Vf)
    plt.plot(t_ax*1e6, shelved)
    plt.plot(t_ax*1e6, shelved2)
    plt.ylabel('Shelved Population')
    plt.xlabel('Shelving time (us)')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
