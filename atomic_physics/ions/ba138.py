""" 138Ba+

TODO: We need a reference for the Einstein A values
A cursory look turned up:
    Kramida, A., et al., NIST Atomic Spectra Database (ver. 5.8), [Online].
However, this data is based of calculations done in the late 60s. These are
only ~10% accurate.

References:
[1] - A. Kramida, NIST Atomic Spectra Database (ver. 5.9) (2021)
[2] - A. A. Madej and J. D. Sankey, Phys. Rev. A 41, 2621, (1990)
[3] - N. Yu, W. Nagourney, and H. Dehmelt, Phys. Rev. Lett. 78, 4898 (1997)
 """
import numpy as np
import scipy.constants as consts
import atomic_physics as ap

# level aliases
ground_level = S12 = ap.Level(n=6, S=1 / 2, L=0, J=1 / 2)
P12 = ap.Level(n=6, S=1 / 2, L=1, J=1 / 2)
P32 = ap.Level(n=6, S=1 / 2, L=1, J=3 / 2)
D32 = ap.Level(n=5, S=1 / 2, L=2, J=3 / 2)
shelf = D52 = ap.Level(n=5, S=1 / 2, L=2, J=5 / 2)


class Ba138(ap.Atom):
    def __init__(self, *, B=None, level_filter=None):
        """88Sr+ atomic structure.

        :param B: B-field in Tesla (can be changed using :meth setB:)
        :param level_filter: list of Levels to include in the simulation, if
            None we include all levels.
        """
        levels = {
            ground_level: ap.LevelData(),
            P12: ap.LevelData(),
            P32: ap.LevelData(),
            D32: ap.LevelData(),
            D52: ap.LevelData(),
        }

        transitions = {
            "493": ap.Transition(
                lower=S12,
                upper=P12,
                A=9.53e7,  # [1]
                freq=2 * np.pi * 607426317510693.9,  # [1]
            ),
            "455": ap.Transition(
                lower=S12,
                upper=P32,
                A=1.11e8,  # [1]
                freq=2 * np.pi * 658116515416903.1,  # [1]
            ),
            "650": ap.Transition(
                lower=D32,
                upper=P12,
                A=3.1e7,  # [1]
                freq=2 * np.pi * 461311910409872.25,  # [1]
            ),
            "585": ap.Transition(
                lower=D32,
                upper=P32,
                A=6.0e6,  # [1]
                freq=2 * np.pi * 512002108316081.56,  # [1]
            ),
            "614": ap.Transition(
                lower=D52,
                upper=P32,
                A=4.12e7,  # [1]
                freq=2 * np.pi * 487990081496342.56,  # [1]
            ),
            "1762": ap.Transition(
                lower=S12,
                upper=D52,
                A=29e-3,  # [2]
                freq=2 * np.pi * 170126433920560.6,  # [1]
            ),
            "2051": ap.Transition(
                lower=S12,
                upper=D32,
                A=12.5e-3,  # [3]
                freq=2 * np.pi * 146114407100821.6,  # [1]
            ),
        }

        super().__init__(
            B=B, I=0, levels=levels, transitions=transitions, level_filter=level_filter
        )
