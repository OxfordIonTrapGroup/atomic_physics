""" Produces Breit-Rabi plots for Calcium 43"""
import numpy as np
import matplotlib.pyplot as plt

from ion_phys import Level
from ion_phys.ions.ca_43_p import Ca43

ion = Ca43()
level = ion.slice(Level(n=4, S=1/2, L=0, J=1/2))
idim = int(np.rint(2*ion.I+1))
jdim = int(np.rint(2*1/2+1))

B_ax = np.arange(0.01, 30000, 200)  # B fields (Gauss)
energies = np.zeros((len(B_ax), idim*jdim))

for idx, B in enumerate(B_ax):
    ion.setB(B*1e-4)
    energies[idx, :] = ion.E[level]

plt.figure()
for idx in range(idim*jdim):
    plt.plot(B_ax, energies[:, idx]/(2*np.pi*1e6))

plt.ylabel('Frequency (MHz)')
plt.xlabel('B field (G)')
plt.grid()
plt.show()
