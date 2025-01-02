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

"""

import numpy as np

from atomic_physics.core import AtomFactory, Level, LevelData, Transition

S12 = Level(n=6, S=1 / 2, L=0, J=1 / 2)
r""" The :math:`\left|n=6, S=1/2, L=0, J=1/2\right>` level.
"""

P12 = Level(n=6, S=1 / 2, L=1, J=1 / 2)
r""" The :math:`\left|n=6, S=1/2, L=1, J=1/2\right>` level."""

P32 = Level(n=6, S=1 / 2, L=1, J=3 / 2)
r""" The :math:`\left|n=6, S=1/2, L=1, J=3/2\right>` level."""

D32 = Level(n=5, S=1 / 2, L=2, J=3 / 2)
r""" The :math:`\left|n=5, S=1/2, L=2, J=3/2\right>` level."""

D52 = Level(n=5, S=1 / 2, L=2, J=5 / 2)
r""" The :math:`\left|n=5, S=1/2, L=2, J=5/2\right>` level."""

ground_level = S12
r""" Alias for the :math:`\left|n=6, S=1/2, L=0, J=1/2\right>` ground level of
:math:`^{138}\mathrm{Ba}^+`.
"""

shelf = D52
r""" Alias for the :math:`\left|n=5, S=1/2, L=2, J=5/2\right>` "shelf" level of
:math:`^{138}\mathrm{Ba}^+`.
"""


level_data = (
    LevelData(level=ground_level, Ahfs=0, Bhfs=0),
    LevelData(level=P12, Ahfs=0, Bhfs=0),
    LevelData(level=P32, Ahfs=0, Bhfs=0),
    LevelData(level=D32, Ahfs=0, Bhfs=0),
    LevelData(level=D52, Ahfs=0, Bhfs=0),
)

transitions = {
    "493": Transition(
        lower=S12,
        upper=P12,
        einstein_A=9.53e7,  # [1]
        frequency=2 * np.pi * 607426317510693.9,  # [1]
    ),
    "455": Transition(
        lower=S12,
        upper=P32,
        einstein_A=1.11e8,  # [1]
        frequency=2 * np.pi * 658116515416903.1,  # [1]
    ),
    "650": Transition(
        lower=D32,
        upper=P12,
        einstein_A=3.1e7,  # [1]
        frequency=2 * np.pi * 461311910409872.25,  # [1]
    ),
    "585": Transition(
        lower=D32,
        upper=P32,
        einstein_A=6.0e6,  # [1]
        frequency=2 * np.pi * 512002108316081.56,  # [1]
    ),
    "614": Transition(
        lower=D52,
        upper=P32,
        einstein_A=4.12e7,  # [1]
        frequency=2 * np.pi * 487990081496342.56,  # [1]
    ),
    "1762": Transition(
        lower=S12,
        upper=D52,
        einstein_A=1 / 30.14,  # [2]
        frequency=2 * np.pi * 170126433920560.6,
    ),
    "2051": Transition(
        lower=S12,
        upper=D32,
        einstein_A=12.5e-3,  # [3]
        frequency=2 * np.pi * 146114407100821.62,
    ),
}

Ba138 = AtomFactory(nuclear_spin=0.0, level_data=level_data, transitions=transitions)
r""" :math:`^{138}\mathrm{Ba}^+` atomic structure. """
