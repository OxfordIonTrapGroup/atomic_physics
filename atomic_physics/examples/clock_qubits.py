import numpy as np

import atomic_physics as ap
import atomic_physics.ions.ca43 as ca43


if __name__ == "__main__":
    # all seems about correct (e.g. agrees with TPH thesis) but expect some
    # numerical inaccuracy, particularly around the second-order field
    # sensitivities. To improve we should add a special case to the derivative
    # calculation that uses the BR formula!
    ion = ca43.Ca43(level_filter=[ca43.ground_level])

    print("Field-independent points:")
    for M3 in range(-3, +3 + 1):
        for q in [-1, 0, 1]:
            ion.setB(1e-4)
            F4 = ion.index(ca43.ground_level, M3 - q, F=4)
            F3 = ion.index(ca43.ground_level, M3, F=3)
            B0 = ap.utils.field_insensitive_point(ion, F4, F3)
            if B0 is not None:
                ion.setB(B0)
                f0 = ion.delta(F4, F3)
                d2fdB2 = ap.utils.d2f_dB2(ion, F4, F3)
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
