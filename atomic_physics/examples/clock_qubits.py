import numpy as np

import atomic_physics.ions.ca43 as ca43
from atomic_physics.utils import d2f_dB2, field_insensitive_point

Ca43 = ca43.Ca43.filter_levels(level_filter=(ca43.ground_level,))

print("Field-independent points:")
for M3 in range(-3, +3 + 1):
    for q in [-1, 0, 1]:
        ion = Ca43(1e-4)
        F4 = ion.get_state_for_F(ca43.ground_level, F=4, M_F=M3 - q)
        F3 = ion.get_state_for_F(ca43.ground_level, F=3, M_F=M3)
        B0 = field_insensitive_point(Ca43, (F4, F3))
        if B0 is not None:
            ion = Ca43(B0)
            f0 = ion.get_transition_frequency_for_states((F4, F3))
            d2fdB2 = d2f_dB2(atom_factory=Ca43, magnetic_field=B0, states=(F4, F3))
            print(
                "4, {} --> 3, {}: {:.6f} GHz @ {:.5f} G ({:.3e} Hz/G^2)".format(
                    M3 - q,
                    M3,
                    f0 / (2 * np.pi * 1e6),
                    B0 * 1e4,
                    d2fdB2 / (2 * np.pi) * 1e-8,
                )
            )
        else:
            print("4, {} --> 3, {}: none found".format(M3 - q, M3))
