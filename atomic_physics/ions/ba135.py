""" 135Ba+

ToDo: We need a reference for the Einstein A values
A cursory look turned up:
    Kramida, A., et al., NIST Atomic Spectra Database (ver. 5.8), [Online].
However, this data is based of calculations done in the late 60s. These are
only ~10% accurate.

TODO: fix isotope shifts

References:
[1] - F. Arbes, et al., Zeitschrift fur Physik D: Atoms, Molecules and
  Clusters, 31, 27 (1994)
[2] - G. Tommaseo, et al., The European Physical Journal D, 25 (2003)
[3] - T. P. Harty, et al. Phys. Rev. Lett 113, 220501 (2014)
[4] - W.  Nortershauser, et al., The European Physical Journal D, 2 (1998)
[5] - J. Benhelm, et al., PHYSICAL REVIEW A 75, 032506 (2007)
[6] - A. Kramida, At. Data Nucl. Data Tables 133-134, 101322 (2020)
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


class Ba135(ap.Atom):
    def __init__(
        self,
        *,
        B: typing.Optional[float] = None,
        level_filter: typing.Optional[typing.List[ap.Level]] = None
    ):
        """135Ba+ atomic structure.

        :param B: B-field in Tesla (can be changed using :meth setB:)
        :param level_filter: list of Levels to include in the simulation, if
            None we include all levels.
        """
        levels = {
            ground_level: ap.LevelData(
                g_J=2.00225664,  # [2]
                g_I=(2 / 3) * 0.838627,  # [3]
                Ahfs=3591.67011718e6 * consts.h,  # [1]
            ),
            P12: ap.LevelData(
                Ahfs=-145.4e6 * consts.h,
                g_I=(2 / 3) * 0.838627,  # [4]  # [3]
            ),
            P32: ap.LevelData(
                Ahfs=-31.4e6 * consts.h,  # [4]
                Bhfs=-6.9 * consts.h,  # [4]
                g_I=(2 / 3) * 0.838627,  # [3]
            ),
            D32: ap.LevelData(
                Ahfs=-47.3e6 * consts.h,  # [4]
                Bhfs=-3.7 * consts.h,  # [4]
                g_I=(2 / 3) * 0.838627,  # [3]
            ),
            D52: ap.LevelData(
                Ahfs=-10.735e6 * consts.h,  # [5]
                Bhfs=38.688e6 * consts.h,  # [5]
                g_I=(2 / 3) * 0.838627,  # [3]
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
