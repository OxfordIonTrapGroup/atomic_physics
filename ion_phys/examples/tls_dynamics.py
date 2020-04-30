""" Simple rate equations example of a 2 level system"""
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from ion_phys.ions.ca43 import Ca43, ground_level, P32
from ion_phys import Laser
from ion_phys.rate_equations import Rates


def main():
    t_ax = np.linspace(0, 0.03e-6, 100)
    I = 1.  # 393 intensity

    ion = Ca43(B=5e-4, level_filter=[ground_level, P32])
    s_stretch = ion.index(ground_level, 4)
    p_stretch = ion.index(P32, +5)

    rates = Rates(ion)
    delta = ion.delta(s_stretch, p_stretch)

    Lasers = [Laser("393", q=+1, I=I, delta=delta)]  # resonant 393 sigma+
    trans = rates.get_transitions(Lasers)

    Vi = np.zeros((ion.num_states, 1))  # initial state
    Vi[s_stretch] = 1  # start in F=4, M=+4
    shelved = np.zeros(len(t_ax))
    for idx, t in np.ndenumerate(t_ax):
        Vf = expm(trans*t)@Vi
        shelved[idx] = sum(Vf[ion.slice(P32)])

    plt.plot(t_ax*1e6, shelved)
    plt.ylabel('P-population')
    plt.xlabel('Pulse duration (us)')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
