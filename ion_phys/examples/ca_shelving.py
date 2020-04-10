""" Simple rate equations example of 393 shelving in 43Ca+. """
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from ion_phys.ions.ca_43_p import Ca43
from ion_phys import Laser, Level
from ion_phys.rate_equations import Rates


def main():
    t_ax = np.linspace(0, 100e-6, 10)
    shelved = np.zeros(len(t_ax))

    ion = Ca43(B=146e-4)
    gl = Level(n=4, S=1/2, L=0, J=1/2)
    shelf = ion.slice(Level(n=3, S=1/2, L=2, J=3/2))
    stretch = ion.index(gl, 4)

    rates = Rates(ion)
    delta = ion.delta(stretch, ion.index(Level(n=4, S=1/2, L=1, J=3/2), +5))
    Lasers = [Laser("393", q=+1, I=0.01, delta=delta)]  # resonant 393 sigma+
    trans = rates.get_tranitions(Lasers)

    for idx, t in np.ndenumerate(t_ax):
        Vi = np.zeros((ion.num_states, 1))  # initial state
        Vi[stretch] = 1  # start in F=4, M=+4
        Vf = expm(trans*t)@Vi
        shelved[idx] = sum(Vf[shelf])

    plt.plot(t_ax*1e6, shelved)
    plt.ylabel('Shelved Population')
    plt.xlabel('Shelving time (us)')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
