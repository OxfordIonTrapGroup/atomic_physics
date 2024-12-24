"""25Mg+

References:
[1] -   W. M. Itano and D. J. Wineland, Precision measurement of the
        ground-state hyperfine constant of Mg+, PRA, 24, 3 (1981)
[2] -   W. H. Yuan et. al., Precision measurement of the light shift of
        25Mg+ions, Phys. Rev. A, 98, 5 (2018)
[3] -   G. Clos et. al., Decoherence-Assisted Spectroscopy of a Single
        Mg+ Ion, Phys. Rev. Lett., 112, 11 (2014)
[4] -   M. Kaur et. al., Radiative transition properties of singly charged
        magnesium, calcium, strontium and barium ions, Atomic Data and Nuclear
        Data Tables, 137 (2021)
[5] -   Z. T. Xu et. al., Precision measurement of the 25Mg+
        ground-state hyperfine constant, Phys. Rev. A, 96, 5, (2017)
[6] -   J. Nguyen, The Linewidth and Hyperfine A Constant of the 2P1/2 State
        of a Magnesium Ion Confined in a Linear Paul Trap, Thesis,
        McMaster University (2009) http://hdl.handle.net/11375/17398
[7] -   N. J. Stone, Table of nuclear magnetic dipole and electric
        quadrupole moments, Atomic Data and Nuclear Data Tables, Volume 90,
        Issue 1 (2005)
[8] -   L. Toppozini, Trapped-Mg+ Apparatus for Control and Structure Studies,
        Thesis, McMaster University (2006) http://hdl.handle.net/11375/21342
"""

import numpy as np
import scipy.constants as consts

from atomic_physics.common import Atom, Level, LevelData, Transition

# level aliases
ground_level = S12 = Level(n=3, S=1 / 2, L=0, J=1 / 2)
P12 = Level(n=3, S=1 / 2, L=1, J=1 / 2)
P32 = Level(n=3, S=1 / 2, L=1, J=3 / 2)


class Mg25(Atom):
    def __init__(
        self,
        *,
        B: float | None = None,
        level_filter: list[Level] | None = None,
    ):
        """25Mg+ atomic structure.

        :param B: B-field (T)
        :param level_filter: list of Levels to include in the simulation, if
            None we include all levels.
        """
        level_data = [
            LevelData(
                level=ground_level,
                g_J=2.002,  # [1] (approximate)
                g_I=(2 / 5) * -0.85545,  # [7]
                Ahfs=-596.2542487e6 * consts.h,  # [5] (or —596.254376(54)e6 [1])
            ),
            LevelData(
                level=P12,
                Ahfs=102.16e6 * consts.h,  # [6]
                g_I=(2 / 5) * -0.85545,  # [7]
            ),
            LevelData(
                level=P32,
                Ahfs=-19.0972e6 * consts.h,  # [8]
                g_I=(2 / 5) * -0.85545,  # [7]
                Bhfs=22.3413e6 * consts.h,  # [8]
            ),
        ]

        transitions = {
            "280": Transition(
                lower=S12,
                upper=P12,
                A=5.58e8,  # [4]
                freq=1069.339957e12 * 2 * np.pi,  # [3]
            ),
            "279": Transition(
                lower=S12,
                upper=P32,
                A=2.60e8,  # [4]
                freq=1072.084547e12
                * 2
                * np.pi,  # [3] (or 1072084547e6 * 2 * np.pi [2])
            ),
        }

        super().__init__(
            B=B,
            I=5 / 2,
            level_data=level_data,
            transitions=transitions,
            level_filter=level_filter,
        )
