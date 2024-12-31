"""Simple rate equations example of 393 shelving in 43Ca+."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

from atomic_physics.core import Laser
from atomic_physics.ions import ca43
from atomic_physics.rate_equations import Rates

t_ax = np.linspace(0, 100e-6, 100)  # Scan the duration of the "shelving" pulse
intensity = 0.02  # 393 intensity

ion = ca43.Ca43(magnetic_field=146e-4)

# Ion starts in the F=4, M=+4 "stretched" state within the 4S1/2 ground-level
stretch = ion.get_state_for_F(ca43.ground_level, F=4, M_F=+4)
Vi = np.zeros((ion.num_states, 1))
Vi[stretch] = 1

# Tune the 393nm laser to resonance with the
# 4S1/2(F=4, M_F=+4) <> 4P3/2(F=5, M_F=+5) transition
detuning = ion.get_transition_frequency_for_states(
    (stretch, ion.get_state_for_F(ca43.P32, F=5, M_F=+5))
)
lasers = (
    Laser("393", polarization=+1, intensity=intensity, detuning=detuning),
)  # resonant 393 sigma+

rates = Rates(ion)
transitions = rates.get_transitions_matrix(lasers)

shelved = np.zeros(len(t_ax))  # Population in the 3D5/2 level at the end
for idx, t in np.ndenumerate(t_ax):
    Vf = expm(transitions * t) @ Vi  # NB use of matrix operations here!
    shelved[idx] = np.sum(Vf[ion.get_slice_for_level(ca43.shelf)])

plt.plot(t_ax * 1e6, shelved)
plt.ylabel("Shelved Population")
plt.xlabel("Shelving time (us)")
plt.grid()
plt.show()
