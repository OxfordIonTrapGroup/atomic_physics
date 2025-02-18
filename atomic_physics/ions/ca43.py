r""":math:`^{43}\mathrm{Ca}^+`

References::

    * [1] F. Arbes, et al., Zeitschrift fur Physik D: Atoms, Molecules and
      Clusters, 31, 27 (1994)
    * [2] G. Tommaseo, et al., The European Physical Journal D, 25 (2003)
    * [3] T. P. Harty, et al. Phys. Rev. Lett 113, 220501 (2014)
    * [4] W.  Nortershauser, et al., The European Physical Journal D, 2 (1998)
    * [5] J. Benhelm, et al., PHYSICAL REVIEW A 75, 032506 (2007)
    * [6] A. Kramida, At. Data Nucl. Data Tables 133-134, 101322 (2020)

"""

import numpy as np
import scipy.constants as consts

from atomic_physics.core import AtomFactory, Level, LevelData, Transition


class Ca43Factory(AtomFactory):
    r"""``AtomFactory`` for :math:`^{43}\mathrm{Ca}^+`.

    Attributes:
        S12: the :math:`\left|n=4, S=1/2, L=0, J=1/2\right>` level.
        P12: the :math:`\left|n=4, S=1/2, L=1, J=1/2\right>` level.
        P32: the :math:`\left|n=4, S=1/2, L=1, J=3/2\right>` level.
        D32: the :math:`\left|n=3, S=1/2, L=2, J=3/2\right>` level.
        D52: the :math:`\left|n=3, S=1/2, L=2, J=5/2\right>` level.
        ground_level: alias for the :math:`\left|n=4, S=1/2, L=0, J=1/2\right>` ground
            level.
        shelf: alias for the :math:`\left|n=3, S=1/2, L=2, J=5/2\right>` "shelf" level.
    """

    S12: Level = Level(n=4, S=1 / 2, L=0, J=1 / 2)
    P12: Level = Level(n=4, S=1 / 2, L=1, J=1 / 2)
    P32: Level = Level(n=4, S=1 / 2, L=1, J=3 / 2)
    D32: Level = Level(n=3, S=1 / 2, L=2, J=3 / 2)
    D52: Level = Level(n=3, S=1 / 2, L=2, J=5 / 2)

    ground_level: Level = S12
    shelf: Level = D52

    def __init__(self):
        level_data = (
            LevelData(
                level=self.S12,
                Ahfs=-3225.60828640e6 * consts.h / 4,  # [1]
                Bhfs=0,
                g_J=2.00225664,  # [2]
                g_I=(2 / 7) * -1.315348,  # [3]
            ),
            LevelData(
                level=self.P12,
                Ahfs=-145.4e6 * consts.h,
                Bhfs=0,
                g_I=(2 / 7) * -1.315348,  # [4]  # [3]
            ),
            LevelData(
                level=self.P32,
                Ahfs=-31.4e6 * consts.h,  # [4]
                Bhfs=-6.9e6 * consts.h,  # [4]
                g_I=(2 / 7) * -1.315348,  # [3]
            ),
            LevelData(
                level=self.D32,
                Ahfs=-47.3e6 * consts.h,  # [4]
                Bhfs=-3.7e6 * consts.h,  # [4]
                g_I=(2 / 7) * -1.315348,  # [3]
            ),
            LevelData(
                level=self.D52,
                Ahfs=-3.8931e6 * consts.h,  # [5]
                Bhfs=-4.241e6 * consts.h,  # [5]
                g_J=1.2003,
                g_I=(2 / 7) * -1.315348,  # [3]
            ),
        )

        transitions = {
            "397": Transition(
                lower=self.S12,
                upper=self.P12,
                einstein_A=132e6,  # [?]
                frequency=2 * np.pi * 755223443.81e6,  # [6]
            ),
            "393": Transition(
                lower=self.S12,
                upper=self.P32,
                einstein_A=135e6,  # [?]
                frequency=2 * np.pi * 761905691.40e6,  # [6]
            ),
            "866": Transition(
                lower=self.D32,
                upper=self.P12,
                einstein_A=8.4e6,  # [?]
                frequency=2 * np.pi * 345996772.78e6,  # [6]
            ),
            "850": Transition(
                lower=self.D32,
                upper=self.P32,
                einstein_A=0.955e6,  # [?]
                frequency=2 * np.pi * 352679020.37e6,  # [6]
            ),
            "854": Transition(
                lower=self.D52,
                upper=self.P32,
                einstein_A=8.48e6,  # [?]
                frequency=350859426.91e6 * 2 * np.pi,  # [6]
            ),
            "729": Transition(
                lower=self.S12,
                upper=self.D52,
                einstein_A=0.856,  # [?]
                frequency=411046264.4881e6 * 2 * np.pi,  # [6]
            ),
            "733": Transition(
                lower=self.S12,
                upper=self.D32,
                einstein_A=0.850,  # [?]#
                frequency=409226671.03e6 * 2 * np.pi,  # [6]
            ),
        }

        super().__init__(
            nuclear_spin=7 / 2, level_data=level_data, transitions=transitions
        )


Ca43 = Ca43Factory()
r""" :math:`^{43}\mathrm{Ca}^+` atomic structure. """
