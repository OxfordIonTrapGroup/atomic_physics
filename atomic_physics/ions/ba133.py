r""":math:`^{133}\mathrm{Ba}^+`

The transition frequencies are calculated based on the :math:`^{138}\mathrm{Ba}^+`
frequencies from [1] and the isotope shifts in the second reference listed next to
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

from atomic_physics.core import AtomFactory, Level, LevelData, Transition


class Ba133Factory(AtomFactory):
    r"""``AtomFactory`` for :math:`^{133}\mathrm{Ba}^+`.

    Attributes:
        S12: the :math:`\left|n=6, S=1/2, L=0, J=1/2\right>` level.
        P12: the :math:`\left|n=6, S=1/2, L=1, J=1/2\right>` level.
        P32: the :math:`\left|n=6, S=1/2, L=1, J=3/2\right>` level.
        D32: the :math:`\left|n=5, S=1/2, L=2, J=3/2\right>` level.
        D52: the :math:`\left|n=5, S=1/2, L=2, J=5/2\right>` level.
        ground_level: alias for the :math:`\left|n=6, S=1/2, L=0, J=1/2\right>` ground
            level.
        shelf: alias for the :math:`\left|n=5, S=1/2, L=2, J=5/2\right>` "shelf" level.
    """

    S12: Level = Level(n=6, S=1 / 2, L=0, J=1 / 2)
    P12: Level = Level(n=6, S=1 / 2, L=1, J=1 / 2)
    P32: Level = Level(n=6, S=1 / 2, L=1, J=3 / 2)
    D32: Level = Level(n=5, S=1 / 2, L=2, J=3 / 2)
    D52: Level = Level(n=5, S=1 / 2, L=2, J=5 / 2)

    ground_level: Level = S12
    shelf: Level = D52

    def __init__(self):
        level_data = (
            LevelData(
                level=self.ground_level,
                Ahfs=-9925.45355459e6 * consts.h,  # [6]
                Bhfs=0,
                g_J=2.0024906,  # [5]
                g_I=(2 / 1) * -0.77167,  # [4]
            ),
            LevelData(
                level=self.P12,
                Ahfs=-1840e6 * consts.h,  # [6]
                Bhfs=0,
                g_I=(2 / 1) * -0.77167,  # [4]
            ),
            LevelData(
                level=self.P32,
                Ahfs=-311.5e6 * consts.h,  # [7]
                Bhfs=0,
                g_I=(2 / 1) * -0.77167,  # [4]
            ),
            LevelData(
                level=self.D32,
                Ahfs=-468.5e6 * consts.h,  # [6]
                Bhfs=0,
                g_I=(2 / 1) * -0.77167,  # [4]
            ),
            LevelData(
                level=self.D52,
                Ahfs=83e6 * consts.h / 3,  # [7]
                Bhfs=0,
                g_I=(2 / 1) * -0.77167,  # [4]
            ),
        )

        transitions = {
            "493": Transition(
                lower=self.S12,
                upper=self.P12,
                einstein_A=9.53e7,  # [1]
                frequency=2 * np.pi * 607426317511066.9,  # [1], [6]
            ),
            "455": Transition(
                lower=self.S12,
                upper=self.P32,
                einstein_A=1.11e8,  # [1]
                frequency=2 * np.pi * 658116515417261.1,  # [1], [7]
            ),
            "650": Transition(
                lower=self.D32,
                upper=self.P12,
                einstein_A=3.1e7,  # [1]
                frequency=2 * np.pi * 461311910410070.25,  # [1], [6]
            ),
            "585": Transition(
                lower=self.D32,
                upper=self.P32,
                einstein_A=6.0e6,  # [1]
                frequency=2 * np.pi * 512002108316264.5,
            ),
            "614": Transition(
                lower=self.D52,
                upper=self.P32,
                einstein_A=4.12e7,  # [1]
                frequency=2 * np.pi * 487990081496558.56,  # [1], [7]
            ),
            "1762": Transition(
                lower=self.S12,
                upper=self.D52,
                einstein_A=1 / 30.14,  # [2]
                frequency=2 * np.pi * 170126433920702.56,
            ),
            "2051": Transition(
                lower=self.S12,
                upper=self.D32,
                einstein_A=12.5e-3,  # [3]
                frequency=2 * np.pi * 146114407100996.62,
            ),
        }

        super().__init__(
            nuclear_spin=1 / 2, level_data=level_data, transitions=transitions
        )


Ba133 = Ba133Factory()
r""" :math:`^{133}\mathrm{Ba}^+` atomic structure. """
