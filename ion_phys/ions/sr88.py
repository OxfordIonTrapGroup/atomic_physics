""" 88Sr+

References:
[1] A. Kramida, NIST Atomic Spectra Database (ver. 5.9) (2021)
[2] P. Dubé, Metrologia (2015)
[3] V. Letchumanan, Phys. Rev. A (2005)
 """
import numpy as np
import scipy.constants as consts
from ion_phys import Level, LevelData, Transition, Ion


# level aliases
ground_level = S12 = Level(n=5, S=1 / 2, L=0, J=1 / 2)
P12 = Level(n=5, S=1 / 2, L=1, J=1 / 2)
P32 = Level(n=5, S=1 / 2, L=1, J=3 / 2)
D32 = Level(n=4, S=1 / 2, L=2, J=3 / 2)
shelf = D52 = Level(n=4, S=1 / 2, L=2, J=5 / 2)


class Sr88(Ion):
    def __init__(self, *, B=None, level_filter=None):
        """88Sr+ atomic structure.

        :param B: B-field in Tesla (can be changed using :meth setB:)
        :param level_filter: list of Levels to include in the simulation, if
            None we include all levels.
        """
        levels = {
            ground_level: LevelData(),
            P12: LevelData(),
            P32: LevelData(),
            D32: LevelData(),
            D52: LevelData(),
        }

        transitions = {
            "422": Transition(
                lower=S12,
                upper=P12,
                A=127.9e6,  # [1]
                freq=2 * np.pi * 711162972859365.0,  # [1]
            ),
            "408": Transition(
                lower=S12,
                upper=P32,
                A=141e6,  # [1]
                freq=2 * np.pi * 735197363032326.0,  # [1]
            ),
            "1092": Transition(
                lower=D32,
                upper=P12,
                A=7.46e6,  # [1]
                freq=2 * np.pi * 274664149123480.0,  # [1]
            ),
            "1004": Transition(
                lower=D32,
                upper=P32,
                A=1.0e6,  # [1]
                freq=2 * np.pi * 298697611773804.0,  # [1]
            ),
            "1033": Transition(
                lower=D52,
                upper=P32,
                A=8.7e6,  # [1]
                freq=2 * np.pi * 290290973185754.0,  # [1]
            ),
            "674": Transition(
                lower=S12,
                upper=D52,
                A=2.55885,  # [3]
                freq=2 * np.pi * 444779044095485.27,  # [2]
            ),
            "687": Transition(
                lower=S12,
                upper=D32,
                A=2.299,  # [1]
                freq=2 * np.pi * 436495331872197.0,  # [1]
            ),
        }

        super().__init__(
            B=B, I=0, levels=levels, transitions=transitions, level_filter=level_filter
        )
