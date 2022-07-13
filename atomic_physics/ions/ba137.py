""" 137Ba+

TODO: We need a reference for the Einstein A values
A cursory look turned up:
    Kramida, A., et al., NIST Atomic Spectra Database (ver. 5.8), [Online].
However, this data is based of calculations done in the late 60s. These are
only ~10% accurate.

TODO: fix isotope shifts

[1] https://journals.aps.org/pr/pdf/10.1103/PhysRev.102.1334
[2] Hucul, David, et al. "Spectroscopy of a synthetic trapped ion qubit." Physical review letters 119.10 (2017): 100501.
    https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.119.100501
[3] - G. Tommaseo, et al., The European Physical Journal D, 25 (2003)
[4] A. Kramida, NIST Atomic Spectra Database (ver. 5.9) (2021)
[5] https://journals.aps.org/pra/pdf/10.1103/PhysRevA.33.2117
"""
import numpy as np
import typing
import scipy.constants as consts
import atomic_physics as ap

# level aliases
ground_level = S12 = ap.Level(n=6, S=1 / 2, L=0, J=1 / 2)
P12 = ap.Level(n=6, S=1 / 2, L=1, J=1 / 2)
P32 = ap.Level(n=6, S=1 / 2, L=1, J=3 / 2)
D32 = ap.Level(n=5, S=1 / 2, L=2, J=3 / 2)
shelf = D52 = ap.Level(n=5, S=1 / 2, L=2, J=5 / 2)


class Ba137(ap.Atom):
    def __init__(
        self,
        *,
        B: typing.Optional[float] = None,
        level_filter: typing.Optional[typing.List[ap.Level]] = None
    ):
        """137Ba+ atomic structure.

        :param B: B-field in Tesla (can be changed using :meth setB:)
        :param level_filter: list of Levels to include in the simulation, if
            None we include all levels.
        """
        levels = {
            ground_level: ap.LevelData(
                g_J=2.00225664,  # [2]
                g_I=(2 / 3) * -0.93107,  # [1]
                Ahfs=4018.87083385e6 * consts.h,  # [2]
            ),
            P12: ap.LevelData(
                Ahfs=743.7e6 * consts.h, # [1]
                g_I=(2 / 3) * -0.93107  # [1]
            ),
            P32: ap.LevelData(
                Ahfs=126.9e6 * consts.h,  # [5]
                Bhfs=92.8e6 * consts.h,  # [5]
                g_I=(2 / 3) * -0.93107,  # [1]
            ),
            D32: ap.LevelData(
                Ahfs=44.5417 * consts.h,  # [2]
                Bhfs=189.7288e6 * consts.h,  # [2]
                g_I=(2 / 3) * -0.93107,  # [1]
            ),
            D52: ap.LevelData(
                Ahfs=-12.028e6 * consts.h,  # [5]
                Bhfs=59.533e6 * consts.h,  # [5]
                g_I=(2 / 3) * -0.93107,  # [1]
            ),
        }

        transitions = {
            "493":
            ap.Transition(
                lower=S12,
                upper=P12,
                A=9.53e7,  # [1]
                freq=2 * np.pi * 607426317510693.9  # [1]
            ),
            "455":
            ap.Transition(
                lower=S12,
                upper=P32,
                A=1.11e8,  # [1]
                freq=2 * np.pi * 658116515416903.1  # [1]
            ),
            "650":
            ap.Transition(
                lower=D32,
                upper=P12,
                A=3.1e7,  # [1]
                freq=2 * np.pi * 461311910409872.25  # [1]
            ),
            "585":
            ap.Transition(
                lower=D32,
                upper=P32,
                A=6.0e6,  # [1]
                freq=2 * np.pi * 512002108316081.56  # [1]
            ),
            "614":
            ap.Transition(
                lower=D52,
                upper=P32,
                A=4.12e7,  # [1]
                freq=2 * np.pi * 487990081496342.56  # [1]
            ),
            "1762":
            ap.Transition(
                lower=S12,
                upper=D52,
                A=0.04,  # [1]
                freq=2 * np.pi * 170126433920560.6  # [1]
            ),
            "2051":
            ap.Transition(
                lower=S12,
                upper=D32,
                A=0.04,  # [1]
                freq=2 * np.pi * 146114407100821.6  # [1]
            ),
        }

        super().__init__(
            B=B,
            I=3 / 2,
            levels=levels,
            transitions=transitions,
            level_filter=level_filter,
        )
