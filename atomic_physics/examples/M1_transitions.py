from atomic_physics.ions import ca43
import scipy.constants as consts

uB = consts.physical_constants["Bohr magneton"][0]


if __name__ == "__main__":
    # all seems correct (e.g. agrees with TPH thesis table E.4)
    ion = ca43.Ca43(level_filter=[ca43.ground_level])
    ion.setB(146.0942e-4)  # magic field for 0->1 clock qubit
    ion.calc_M1()

    R = ion.M1 / uB

    for M3 in range(-3, +3 + 1):
        for q in [-1, 0, 1]:
            F4 = ion.index(ca43.ground_level, M3 - q, F=4)
            F3 = ion.index(ca43.ground_level, M3, F=3)
            Rnm = R[
                ion.index(ca43.ground_level, M=M3, F=3),
                ion.index(ca43.ground_level, M=M3 - q, F=4),
            ]
            print("Rnm for F=3, M={} -> F=4," "M={}: {:.6f}".format(M3, M3 - q, Rnm))
