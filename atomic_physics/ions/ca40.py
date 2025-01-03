r""":math:`^{40}\mathrm{Ca}^+`

References::

    * [1] A. Kramida, At. Data Nucl. Data Tables 133-134, 101322 (2020)
    * [2] T. P. Harty, DPhil Thesis (2013)
    * [3] M. Chwalla et all, PRL 102, 023002 (2009)

"""

import numpy as np

from atomic_physics.core import AtomFactory, Level, LevelData, Transition

# level aliases
ground_level = S12 = Level
P12 = Level
P32 = Level
D32 = Level
shelf = D52 = Level


S12 = Level(n=4, S=1 / 2, L=0, J=1 / 2)
r""" The :math:`\left|n=4, S=1/2, L=0, J=1/2\right>` level.
"""

P12 = Level(n=4, S=1 / 2, L=1, J=1 / 2)
r""" The :math:`\left|n=4, S=1/2, L=1, J=1/2\right>` level."""

P32 = Level(n=4, S=1 / 2, L=1, J=3 / 2)
r""" The :math:`\left|n=4, S=1/2, L=1, J=3/2\right>` level."""

D32 = Level(n=3, S=1 / 2, L=2, J=3 / 2)
r""" The :math:`\left|n=3, S=1/2, L=2, J=3/2\right>` level."""

D52 = Level(n=3, S=1 / 2, L=2, J=5 / 2)
r""" The :math:`\left|n=3, S=1/2, L=2, J=5/2\right>` level."""

ground_level = S12
r""" Alias for the :math:`\left|n=4, S=1/2, L=0, J=1/2\right>` ground level of
:math:`^{40}\mathrm{Ca}^+`.
"""

shelf = D52
r""" Alias for the :math:`\left|n=3, S=1/2, L=2, J=5/2\right>` "shelf" level of
:math:`^{40}\mathrm{Ca}^+`.
"""


level_data = (
    LevelData(level=ground_level, g_J=2.00225664, Ahfs=0, Bhfs=0),  # [2]
    LevelData(level=P12, Ahfs=0, Bhfs=0),
    LevelData(level=P32, Ahfs=0, Bhfs=0),
    LevelData(level=D32, Ahfs=0, Bhfs=0),
    LevelData(level=D52, g_J=1.2003340, Ahfs=0, Bhfs=0),  # [3]
)

transitions = {
    "397": Transition(
        lower=S12,
        upper=P12,
        einstein_A=132e6,  # [?]
        frequency=2 * np.pi * 755222765771e3,  # [1]
    ),
    "393": Transition(
        lower=S12,
        upper=P32,
        einstein_A=135e6,  # [?]
        frequency=2 * np.pi * 761905012599e3,  # [1]
    ),
    "866": Transition(
        lower=D32,
        upper=P12,
        einstein_A=8.4e6,  # [?]
        frequency=2 * np.pi * 346000235016e3,  # [1]
    ),
    "850": Transition(
        lower=D32,
        upper=P32,
        einstein_A=0.955e6,  # [?]
        frequency=2 * np.pi * 352682481844e3,  # [1]
    ),
    "854": Transition(
        lower=D52,
        upper=P32,
        einstein_A=8.48e6,  # [?]
        frequency=2 * np.pi * 350862882823e3,  # [1]
    ),
    "729": Transition(
        lower=S12,
        upper=D52,
        einstein_A=0.856,  # [?]
        frequency=411042129776.4017e3 * 2 * np.pi,  # [1]
    ),
    "733": Transition(
        lower=S12,
        upper=D32,
        einstein_A=0.850,  # [?]
        frequency=409222530754.868e3 * 2 * np.pi,  # [1]
    ),
}

Ca40 = AtomFactory(nuclear_spin=0.0, level_data=level_data, transitions=transitions)
r""" :math:`^{40}\mathrm{Ca}^+` atomic structure. """
