import numpy as np
import scipy.constants as consts
import matplotlib.pyplot as plt

import ion_phys.common as ip
from ion_phys.common import Level
from ion_phys.atoms.ca_43_p import atom

bField = np.arange(0.01, 300, 2)
energy_g = []

for B in bField:
    ip.init(atom, B*1e-4)

    levels = atom["levels"]
    ground_level = levels[Level(n=3, S=1/2, L=2, J=3/2)]
    J = ground_level["MJ"][-1]
    M_g = ground_level["M"]
    F_g = ground_level["F"]
    E_g = ground_level["E"]

    F_list = np.arange(atom["I"] - J, atom["I"] + J + 1)
    for F in F_list:
        for M in np.arange(-F, F + 1):
            energy_g.append(E_g[np.logical_and(F_g == F, M_g == M)]/consts.h)

plt.figure()
idx = np.int(np.sum(2*F_list) + len(F_list))
for i in range(idx):
    plt.plot(bField, np.array(energy_g[i::idx])/1e6, color='k')

plt.ylabel('Frequency (MHz)')
plt.xlabel('B field (G)')
plt.show()
