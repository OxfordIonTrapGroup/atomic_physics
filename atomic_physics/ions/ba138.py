r""":math:`^{138}\mathrm{Ba}^+`

Where no references are given next to the transition frequencies, they
were calculated based on transition frequencies between other levels.
For example the frequency of the 1762 nm transition f(1762) was found
from f(1762)=f(455)-f(614).

References::

    * [1] A. Kramida, NIST Atomic Spectra Database (ver. 5.9) (2021)
    * [2] Zhiqiang Zhang, K. J. Arnold, S. R. Chanu, R. Kaewuam,
      M. S. Safronova, and M. D. Barrett Phys. Rev. A 101, 062515 (2020)
    * [3] N. Yu, W. Nagourney, and H. Dehmelt, Phys. Rev. Lett. 78, 4898 (1997)
    * [4] K. H. Knoll et al., PRA54 1199 (1996)
    * [5] O. Poulson & P.J. Ramanujam, PRA 14 1463 (1976)
    * [6] N. Kurz et al., PRA 82 030501 (2010)

"""

import numpy as np

from atomic_physics.core import AtomFactory, Level, LevelData, Transition


class Ba138Factory(AtomFactory):
    r""":class:`~atomic_physics.core.AtomFactory` for :math:`^{138}\mathrm{Ba}^+`.

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
            LevelData(level=self.S12, g_J =  2.0024922, Ahfs=0, Bhfs=0), # [4]
            LevelData(level=self.P12, g_J = 0.672, Ahfs=0, Bhfs=0), # [5]
            LevelData(level=self.P32, g_J = 1.328, Ahfs=0, Bhfs=0), # [5]
            LevelData(level=self.D32, g_J = 0.7993278, Ahfs=0, Bhfs=0), # [4]
            LevelData(level=self.D52, g_J = 1.2020, Ahfs=0, Bhfs=0), # [6]
        )

        transitions = {
            "493": Transition(
                lower=self.S12,
                upper=self.P12,
                einstein_A=9.53e7,  # [1]
                frequency=2 * np.pi * 607426317510693.9,  # [1]
            ),
            "455": Transition(
                lower=self.S12,
                upper=self.P32,
                einstein_A=1.11e8,  # [1]
                frequency=2 * np.pi * 658116515416903.1,  # [1]
            ),
            "650": Transition(
                lower=self.D32,
                upper=self.P12,
                einstein_A=3.1e7,  # [1]
                frequency=2 * np.pi * 461311910409872.25,  # [1]
            ),
            "585": Transition(
                lower=self.D32,
                upper=self.P32,
                einstein_A=6.0e6,  # [1]
                frequency=2 * np.pi * 512002108316081.56,  # [1]
            ),
            "614": Transition(
                lower=self.D52,
                upper=self.P32,
                einstein_A=4.12e7,  # [1]
                frequency=2 * np.pi * 487990081496342.56,  # [1]
            ),
            "1762": Transition(
                lower=self.S12,
                upper=self.D52,
                einstein_A=1 / 30.14,  # [2]
                frequency=2 * np.pi * 170126433920560.6,
            ),
            "2051": Transition(
                lower=self.S12,
                upper=self.D32,
                einstein_A=12.5e-3,  # [3]
                frequency=2 * np.pi * 146114407100821.62,
            ),
        }

        super().__init__(
            nuclear_spin=0.0, level_data=level_data, transitions=transitions
        )


Ba138 = Ba138Factory()
r""" :math:`^{138}\mathrm{Ba}^+` atomic structure. """
