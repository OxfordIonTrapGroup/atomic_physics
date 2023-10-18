""" 43Ca+

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
ground_level = S12 = ap.Level(n=4, S=1 / 2, L=0, J=1 / 2)
P12 = ap.Level(n=4, S=1 / 2, L=1, J=1 / 2)
P32 = ap.Level(n=4, S=1 / 2, L=1, J=3 / 2)
D32 = ap.Level(n=4, S=1 / 2, L=2, J=3 / 2)
shelf = D52 = ap.Level(n=4, S=1 / 2, L=2, J=5 / 2)


class Ca43(ap.Atom):
    def __init__(
        self,
        *,
        B: typing.Optional[float] = None,
        level_filter: typing.Optional[typing.List[ap.Level]] = None
    ):
        """43Ca+ atomic structure.

        :param B: B-field in Tesla (can be changed using :meth setB:)
        :param level_filter: list of Levels to include in the simulation, if
            None we include all levels.
        """
        levels = {
            ground_level: ap.LevelData(
                g_J=2.00225664,  # [2]
                g_I=(2 / 7) * -1.315348,  # [3]
                Ahfs=-3225.60828640e6 * consts.h / 4,  # [1]
            ),
            P12: ap.LevelData(
                Ahfs=-145.4e6 * consts.h, g_I=(2 / 7) * -1.315348  # [4]  # [3]
            ),
            P32: ap.LevelData(
                Ahfs=-31.4e6 * consts.h,  # [4]
                Bhfs=-6.9e6 * consts.h,  # [4]
                g_I=(2 / 7) * -1.315348,  # [3]
            ),
            D32: ap.LevelData(
                Ahfs=-47.3e6 * consts.h,  # [4]
                Bhfs=-3.7e6 * consts.h,  # [4]
                g_I=(2 / 7) * -1.315348,  # [3]
            ),
            D52: ap.LevelData(
                Ahfs=-3.8931e6 * consts.h,  # [5]
                Bhfs=-4.241e6 * consts.h,  # [5]
                g_I=(2 / 7) * -1.315348,  # [3]
            ),
        }

        transitions = {
            "397": ap.Transition(
                lower=S12,
                upper=P12,
                A=132e6,  # [?]
                freq=2 * np.pi * 755223443.81e6,  # [6]
            ),
            "393": ap.Transition(
                lower=S12,
                upper=P32,
                A=135e6,  # [?]
                freq=2 * np.pi * 761905691.40e6,  # [6]
            ),
            "866": ap.Transition(
                lower=D32,
                upper=P12,
                A=8.4e6,  # [?]
                freq=2 * np.pi * 345996772.78e6,  # [6]
            ),
            "850": ap.Transition(
                lower=D32,
                upper=P32,
                A=0.955e6,  # [?]
                freq=2 * np.pi * 352679020.37e6,  # [6]
            ),
            "854": ap.Transition(
                lower=D52,
                upper=P32,
                A=8.48e6,  # [?]
                freq=2 * np.pi * 350859426.91e6,  # [6]
            ),
            "729": ap.Transition(
                lower=S12,
                upper=D52,
                A=0.856,  # [?]
                freq=411046264.4881 * 2 * np.pi,  # [6]
            ),
            "733": ap.Transition(
                lower=S12,
                upper=D32,
                A=0.850,  # [?]
                freq=409226671.03 * 2 * np.pi,  # [6]
            ),
        }

        super().__init__(
            B=B,
            I=7 / 2,
            levels=levels,
            transitions=transitions,
            level_filter=level_filter,
        )
