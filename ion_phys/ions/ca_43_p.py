""" 43Ca+

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


class Ca43(Ion):
    def __init__(self, B=None):
        levels = {
            Level(n=4, L=0, S=1/2, J=1/2): LevelData(
                g_J=2.00225664,  # [2]
                g_I=(2 / 7) * -1.315348,  # [3]
                Ahfs=-3225.60828640e6 * consts.h / 4  # [1]
            ),
            Level(n=4, L=1, S=1/2, J=1/2): LevelData(
                Ahfs=-145.4e6 * consts.h,  # [4]
                g_I=(2 / 7) * -1.315348  # [3]
            ),
            Level(n=4, L=1, S=1/2, J=3/2): LevelData(
                Ahfs=-31.4e6 * consts.h,  # [4]
                Bhfs=-6.9 * consts.h,  # [4]
                g_I=(2 / 7) * -1.315348  # [3]
            ),
            Level(n=3, L=2, S=1/2, J=3/2): LevelData(
                Ahfs=-47.3e6 * consts.h,  # [4]
                Bhfs=-3.7 * consts.h,  # [4]
                g_I=(2 / 7) * -1.315348  # [3]
            ),
            Level(n=3, L=2, S=1/2, J=5/2): LevelData(
                Ahfs=-3.8931e6 * consts.h,  # [5]
                Bhfs=4.241 * consts.h,  # [5]
                g_I=(2 / 7) * -1.315348  # [3]
            )
        }

        transitions = {
            "397": Transition(
                lower=Level(n=4, L=0, S=1/2, J=1/2),
                upper=Level(n=4, L=1, S=1/2, J=1/2),
                A=132e6,  # [?]
                freq=2*np.pi*755.2227662e12  # [?]
            ),
            "393": Transition(
                lower=Level(n=4, L=0, S=1/2, J=1/2),
                upper=Level(n=4, L=1, S=1/2, J=3/2),
                A=132e6,  # [?]
                freq=2*np.pi*consts.c/393e-9  # [?]
            ),
            "866": Transition(
                lower=Level(n=3, L=2, S=1/2, J=3/2),
                upper=Level(n=4, L=1, S=1/2, J=1/2),
                A=0,  # [?]
                freq=2*np.pi*consts.c/866e-9  # [?]
            ),
            "854": Transition(
                lower=Level(n=3, L=2, S=1/2, J=5/2),
                upper=Level(n=4, L=1, S=1/2, J=3/2),
                A=0,  # [?]
                freq=2*np.pi*consts.c/854e-9  # [?]
            ),
        }

        super().__init__(B, I=7/2, levels=levels, transitions=transitions)
