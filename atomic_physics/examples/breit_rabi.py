"""Produces Breit-Rabi plots for Calcium 43"""

import matplotlib.pyplot as plt
import numpy as np

from atomic_physics.ions.ca43 import Ca43

factory = Ca43.filter_levels(level_filter=(Ca43.ground_level,))
idim = int(np.rint(2 * Ca43.nuclear_spin + 1))
jdim = int(np.rint(2 * 1 / 2 + 1))

field_ax = np.arange(0.01, 30000, 200)  # B fields (Gauss)
energies = np.zeros((len(field_ax), idim * jdim))

for idx, magnetic_field in enumerate(field_ax):
    ion = factory(magnetic_field * 1e-4)
    energies[idx, :] = ion.state_energies

plt.figure()
for idx in range(idim * jdim):
    plt.plot(field_ax, energies[:, idx] / (2 * np.pi * 1e6))

plt.ylabel("Frequency (MHz)")
plt.xlabel("B field (G)")
plt.grid()
plt.show()
