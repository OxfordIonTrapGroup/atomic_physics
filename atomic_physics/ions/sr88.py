r""":math:`^{88}\mathrm{Sr}^+`

References::

   * [1] A. Kramida, NIST Atomic Spectra Database (ver. 5.9) (2021)
   * [2] P. Dubé, Metrologia (2015)
   * [3] V. Letchumanan, Phys. Rev. A (2005)

"""

import numpy as np

from atomic_physics.core import AtomFactory, Level, LevelData, Transition


class Sr88Factory(AtomFactory):
    r""":class:`~atomic_physics.core.AtomFactory` for :math:`^{88}\mathrm{Sr}^+`.

    Attributes:
        S12: the :math:`\left|n=5, S=1/2, L=0, J=1/2\right>` level.
        P12: the :math:`\left|n=5, S=1/2, L=1, J=1/2\right>` level.
        P32: the :math:`\left|n=5, S=1/2, L=1, J=3/2\right>` level.
        D32: the :math:`\left|n=4, S=1/2, L=2, J=3/2\right>` level.
        D52: the :math:`\left|n=4, S=1/2, L=2, J=5/2\right>` level.

        ground_level: alias for the :math:`\left|n=5, S=1/2, L=0, J=1/2\right>` ground
            level.
        shelf: alias for the :math:`\left|n=4, S=1/2, L=2, J=5/2\right>` "shelf" level.
    """

    S12: Level = Level(n=5, S=1 / 2, L=0, J=1 / 2)
    P12: Level = Level(n=5, S=1 / 2, L=1, J=1 / 2)
    P32: Level = Level(n=5, S=1 / 2, L=1, J=3 / 2)
    D32: Level = Level(n=4, S=1 / 2, L=2, J=3 / 2)
    D52: Level = Level(n=4, S=1 / 2, L=2, J=5 / 2)

    ground_level: Level = S12
    shelf: Level = D52

    def __init__(self):
        level_data = (
            LevelData(level=self.S12, Ahfs=0, Bhfs=0),
            LevelData(level=self.P12, Ahfs=0, Bhfs=0),
            LevelData(level=self.P32, Ahfs=0, Bhfs=0),
            LevelData(level=self.D32, Ahfs=0, Bhfs=0),
            LevelData(level=self.D52, Ahfs=0, Bhfs=0),
        )

        transitions = {
            "422": Transition(
                lower=self.S12,
                upper=self.P12,
                einstein_A=127.9e6,  # [1]
                frequency=2 * np.pi * 711162972859365.0,  # [1]
            ),
            "408": Transition(
                lower=self.S12,
                upper=self.P32,
                einstein_A=141e6,  # [1]
                frequency=2 * np.pi * 735197363032326.0,  # [1]
            ),
            "1092": Transition(
                lower=self.D32,
                upper=self.P12,
                einstein_A=7.46e6,  # [1]
                frequency=2 * np.pi * 274664149123480.0,  # [1]
            ),
            "1004": Transition(
                lower=self.D32,
                upper=self.P32,
                einstein_A=1.0e6,  # [1]
                frequency=2 * np.pi * 298697611773804.0,  # [1]
            ),
            "1033": Transition(
                lower=self.D52,
                upper=self.P32,
                einstein_A=8.7e6,  # [1]
                frequency=2 * np.pi * 290290973185754.0,  # [1]
            ),
            "674": Transition(
                lower=self.S12,
                upper=self.D52,
                einstein_A=2.55885,  # [3]
                frequency=2 * np.pi * 444779044095485.27,  # [2]
            ),
            "687": Transition(
                lower=self.S12,
                upper=self.D32,
                einstein_A=2.299,  # [1]
                frequency=2 * np.pi * 436495331872197.0,  # [1]
            ),
        }

        super().__init__(
            nuclear_spin=0.0, level_data=level_data, transitions=transitions
        )


Sr88 = Sr88Factory()
r""" :math:`^{88}\mathrm{Sr}^+` atomic structure. """
