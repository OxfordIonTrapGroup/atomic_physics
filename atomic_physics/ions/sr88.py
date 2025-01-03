r""":math:`^{88}\mathrm{Sr}^+`

References::

   * [1] A. Kramida, NIST Atomic Spectra Database (ver. 5.9) (2021)
   * [2] P. Dubé, Metrologia (2015)
   * [3] V. Letchumanan, Phys. Rev. A (2005)

"""

import numpy as np

from atomic_physics.core import AtomFactory, Level, LevelData, Transition

# level aliases
ground_level = S12 = Level
P12 = Level
P32 = Level
D32 = Level
shelf = D52 = Level


S12 = Level(n=5, S=1 / 2, L=0, J=1 / 2)
r""" The :math:`\left|n=5, S=1/2, L=0, J=1/2\right>` level.
"""

P12 = Level(n=5, S=1 / 2, L=1, J=1 / 2)
r""" The :math:`\left|n=5, S=1/2, L=1, J=1/2\right>` level."""

P32 = Level(n=5, S=1 / 2, L=1, J=3 / 2)
r""" The :math:`\left|n=5, S=1/2, L=1, J=3/2\right>` level."""

D32 = Level(n=5, S=1 / 2, L=2, J=3 / 2)
r""" The :math:`\left|n=4, S=1/2, L=2, J=3/2\right>` level."""

D52 = Level(n=4, S=1 / 2, L=2, J=5 / 2)
r""" The :math:`\left|n=4, S=1/2, L=2, J=5/2\right>` level."""

ground_level = S12
r""" Alias for the :math:`\left|n=5, S=1/2, L=0, J=1/2\right>` ground level of
:math:`^{88}\mathrm{Sr}^+`.
"""

shelf = D52
r""" Alias for the :math:`\left|n=4, S=1/2, L=2, J=5/2\right>` "shelf" level of
:math:`^{88}\mathrm{Sr}^+`.
"""

level_data = (
    LevelData(level=ground_level, Ahfs=0, Bhfs=0),
    LevelData(level=P12, Ahfs=0, Bhfs=0),
    LevelData(level=P32, Ahfs=0, Bhfs=0),
    LevelData(level=D32, Ahfs=0, Bhfs=0),
    LevelData(level=D52, Ahfs=0, Bhfs=0),
)

transitions = {
    "422": Transition(
        lower=S12,
        upper=P12,
        einstein_A=127.9e6,  # [1]
        frequency=2 * np.pi * 711162972859365.0,  # [1]
    ),
    "408": Transition(
        lower=S12,
        upper=P32,
        einstein_A=141e6,  # [1]
        frequency=2 * np.pi * 735197363032326.0,  # [1]
    ),
    "1092": Transition(
        lower=D32,
        upper=P12,
        einstein_A=7.46e6,  # [1]
        frequency=2 * np.pi * 274664149123480.0,  # [1]
    ),
    "1004": Transition(
        lower=D32,
        upper=P32,
        einstein_A=1.0e6,  # [1]
        frequency=2 * np.pi * 298697611773804.0,  # [1]
    ),
    "1033": Transition(
        lower=D52,
        upper=P32,
        einstein_A=8.7e6,  # [1]
        frequency=2 * np.pi * 290290973185754.0,  # [1]
    ),
    "674": Transition(
        lower=S12,
        upper=D52,
        einstein_A=2.55885,  # [3]
        frequency=2 * np.pi * 444779044095485.27,  # [2]
    ),
    "687": Transition(
        lower=S12,
        upper=D32,
        einstein_A=2.299,  # [1]
        frequency=2 * np.pi * 436495331872197.0,  # [1]
    ),
}

Sr88 = AtomFactory(nuclear_spin=0, level_data=level_data, transitions=transitions)
r""" :math:`^{88}\mathrm{Sr}^+` atomic structure. """
