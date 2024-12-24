"""133Ba+

The transition frequencies are calculated based on the 138Ba+ frequencies
from [1] and the isotope shifts in the second reference listed next to
the frequency. Where no references are given, the transition frequencies
were calculated based on transition frequencies between other levels.
For example the frequency of the 1762 nm transition f(1762) was found
from f(1762)=f(455)-f(614).

References::

    * [1] A. Kramida, NIST Atomic Spectra Database (ver. 5.9) (2021)
    * [2] Zhiqiang Zhang, K. J. Arnold, S. R. Chanu, R. Kaewuam,
      M. S. Safronova, and M. D. Barrett Phys. Rev. A 101, 062515 (2020)
    * [3] N. Yu, W. Nagourney, and H. Dehmelt, Phys. Rev. Lett. 78, 4898 (1997)
    * [4] N. J. Stone, Table of nuclear magnetic dipole and electric
      quadrupole moments, Atomic Data and Nuclear Data Tables, Volume 90, Issue 1 (2005)
    * [5] H. Knab, K. H. Knöll, F. Scheerer and G. Werth, Zeitschrift für
      Physik D Atoms, Molecules and Clusters volume 25, pages205–208 (1993)
    * [6] David Hucul, Justin E. Christensen, Eric R. Hudson, and
      Wesley C. Campbell, Phys. Rev. Lett. 119, 100501 (2017)
    * [7] - J.E. Christensen, D. Hucul, W.C. Campbell et al., npj Quantum Inf 6, 35
      (2020).

"""

import numpy as np
import scipy.constants as consts

from atomic_physics.common import Atom, Level, LevelData, Transition

# level aliases
ground_level = S12 = Level(n=6, S=1 / 2, L=0, J=1 / 2)
P12 = Level(n=6, S=1 / 2, L=1, J=1 / 2)
P32 = Level(n=6, S=1 / 2, L=1, J=3 / 2)
D32 = Level(n=5, S=1 / 2, L=2, J=3 / 2)
shelf = D52 = Level(n=5, S=1 / 2, L=2, J=5 / 2)


class Ba133(Atom):
    def __init__(
        self,
        *,
        B: float | None = None,
        level_filter: list[Level] | None = None,
    ):
        """133Ba+ atomic structure.

        :param B: B-field (T)
        :param level_filter: list of Levels to include in the simulation, if
            None we include all levels.
        """
        level_data = [
            LevelData(
                level=ground_level,
                Ahfs=-9925.45355459e6 * consts.h,  # [6]
                g_J=2.0024906,  # [5]
                g_I=(2 / 1) * -0.77167,  # [4]
            ),
            LevelData(
                level=P12,
                Ahfs=-1840e6 * consts.h,  # [6]
                g_I=(2 / 1) * -0.77167,  # [4]
            ),
            LevelData(
                level=P32,
                Ahfs=-311.5e6 * consts.h,  # [7]
                g_I=(2 / 1) * -0.77167,  # [4]
            ),
            LevelData(
                level=D32,
                Ahfs=-468.5e6 * consts.h,  # [6]
                g_I=(2 / 1) * -0.77167,  # [4]
            ),
            LevelData(
                level=D52,
                Ahfs=83e6 * consts.h / 3,  # [7]
                g_I=(2 / 1) * -0.77167,  # [4]
            ),
        ]

        transitions = {
            "493": Transition(
                lower=S12,
                upper=P12,
                A=9.53e7,  # [1]
                freq=2 * np.pi * 607426317511066.9,  # [1], [6]
            ),
            "455": Transition(
                lower=S12,
                upper=P32,
                A=1.11e8,  # [1]
                freq=2 * np.pi * 658116515417261.1,  # [1], [7]
            ),
            "650": Transition(
                lower=D32,
                upper=P12,
                A=3.1e7,  # [1]
                freq=2 * np.pi * 461311910410070.25,  # [1], [6]
            ),
            "585": Transition(
                lower=D32,
                upper=P32,
                A=6.0e6,  # [1]
                freq=2 * np.pi * 512002108316264.5,
            ),
            "614": Transition(
                lower=D52,
                upper=P32,
                A=4.12e7,  # [1]
                freq=2 * np.pi * 487990081496558.56,  # [1], [7]
            ),
            "1762": Transition(
                lower=S12,
                upper=D52,
                A=1 / 30.14,  # [2]
                freq=2 * np.pi * 170126433920702.56,
            ),
            "2051": Transition(
                lower=S12,
                upper=D32,
                A=12.5e-3,  # [3]
                freq=2 * np.pi * 146114407100996.62,
            ),
        }

        super().__init__(
            B=B,
            I=1 / 2,
            level_data=level_data,
            transitions=transitions,
            level_filter=level_filter,
        )
