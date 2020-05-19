""" 43Ca+

To do: check numbers and add references. NB the transition frequencies aren't
correct as they don't include isotope shifts!

References:
[1] - F. Arbes, et al., Zeitschrift fur Physik D: Atoms, Molecules and
  Clusters, 31, 27 (1994)
[2] - G. Tommaseo, et al., The European Physical Journal D, 25 (2003)
[3] - T. P. Harty, et al. Phys. Rev. Lett 113, 220501 (2014)
[4] - W.  Nortershauser, et al., The European Physical Journal D, 2 (1998)
[5] - J. Benhelm, et al., PHYSICAL REVIEW A 75, 032506 (2007)
"""
import numpy as np
import scipy.constants as consts
from ion_phys import Level, LevelData, Transition, Ion


# level aliases
ground_level = S12 = Level(n=4, S=1/2, L=0, J=1/2)
P12 = Level(n=4, S=1/2, L=1, J=1/2)
P32 = Level(n=4, S=1/2, L=1, J=3/2)
D32 = Level(n=4, S=1/2, L=2, J=3/2)
shelf = D52 = Level(n=4, S=1/2, L=2, J=5/2)


class Ca43(Ion):
    def __init__(self, *, B=None, level_filter=None):
        """ 43Ca+ atomic structure.

        :param B: B-field in Tesla (can be changed using :meth setB:)
        :param level_filter: list of Levels to include in the simulation, if
            None we include all levels.
        """
        levels = {
            ground_level: LevelData(
                g_J=2.00225664,  # [2]
                g_I=(2 / 7) * -1.315348,  # [3]
                Ahfs=-3225.60828640e6 * consts.h / 4  # [1]
            ),
            P12: LevelData(
                Ahfs=-145.4e6 * consts.h,  # [4]
                g_I=(2 / 7) * -1.315348  # [3]
            ),
            P32: LevelData(
                Ahfs=-31.4e6 * consts.h,  # [4]
                Bhfs=-6.9 * consts.h,  # [4]
                g_I=(2 / 7) * -1.315348  # [3]
            ),
            D32: LevelData(
                Ahfs=-47.3e6 * consts.h,  # [4]
                Bhfs=-3.7 * consts.h,  # [4]
                g_I=(2 / 7) * -1.315348  # [3]
            ),
            D52: LevelData(
                Ahfs=-3.8931e6 * consts.h,  # [5]
                Bhfs=4.241 * consts.h,  # [5]
                g_I=(2 / 7) * -1.315348  # [3]
            )
        }

        transitions = {
            "397": Transition(
                lower=S12,
                upper=P12,
                A=132e6,  # [?]
                freq=2*np.pi*755.2227662e12  # [?]
            ),
            "393": Transition(
                lower=S12,
                upper=P32,
                A=135e6,  # [?]
                freq=2*np.pi*761.9050127e12  # [?]
            ),
            "866": Transition(
                lower=D32,
                upper=P12,
                A=8.4e6,  # [?]
                freq=2*np.pi*consts.c/866e-9  # [?]
            ),
            "850": Transition(
                lower=D32,
                upper=P32,
                A=0.955e6,  # [?]
                freq=2*np.pi*consts.c/850e-9  # [?]
            ),
            "854": Transition(
                lower=D52,
                upper=P32,
                A=8.48e6,  # [?]
                freq=2*np.pi*consts.c/854e-9  # [?]
            ),
            "729": Transition(
                lower=S12,
                upper=D52,
                A=0.856,  # [?]
                freq=411.0421297763932e12*2*np.pi  # [?]
            ),
            "733": Transition(
                lower=S12,
                upper=D32,
                A=0.850,  # [?]
                freq=409.222e12*2*np.pi  # [?]
            ),
        }

        super().__init__(B=B, I=7/2, levels=levels, transitions=transitions,
                         level_filter=level_filter)
