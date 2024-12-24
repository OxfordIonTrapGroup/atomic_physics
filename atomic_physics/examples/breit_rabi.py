"""Produces Breit-Rabi plots for Calcium 43"""

import matplotlib.pyplot as plt
import numpy as np

from atomic_physics.ions import ca43

ion = ca43.Ca43(level_filter=[ca43.ground_level])
idim = int(np.rint(2 * ion.I + 1))
jdim = int(np.rint(2 * 1 / 2 + 1))

B_ax = np.arange(0.01, 30000, 200)  # B fields (Gauss)
energies = np.zeros((len(B_ax), idim * jdim))

for idx, B in enumerate(B_ax):
    ion.setB(B * 1e-4)
    energies[idx, :] = ion.E

plt.figure()
for idx in range(idim * jdim):
    plt.plot(B_ax, energies[:, idx] / (2 * np.pi * 1e6))

plt.ylabel("Frequency (MHz)")
plt.xlabel("B field (G)")
plt.grid()
plt.show()
print("done")
