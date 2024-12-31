import scipy.constants as consts

from atomic_physics.ions import ca43

uB = consts.physical_constants["Bohr magneton"][0]


ion = ca43.Ca43.filter_levels(level_filter=(ca43.ground_level,))(
    magnetic_field=146.0942e-4
)

R = ion.get_magnetic_dipoles() / uB

for M3 in range(-3, +3 + 1):
    for q in [-1, 0, 1]:
        F4 = ion.get_state_for_F(ca43.ground_level, F=4, M_F=M3 - q)
        F3 = ion.get_state_for_F(ca43.ground_level, F=3, M_F=M3)
        Rnm = R[
            ion.get_state_for_F(ca43.ground_level, F=3, M_F=M3),
            ion.get_state_for_F(ca43.ground_level, F=4, M_F=M3 - q),
        ]
        print("Rnm for F=3, M={} -> F=4," "M={}: {:.6f}".format(M3, M3 - q, Rnm))
