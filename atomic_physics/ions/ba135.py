""" 135Ba+

The transition frequencies are calculated based on the 138Ba+ frequencies
from [1] and the isotope shifts in the second reference listed next to
the frequency. Where no references are given, the transition frequencies
were calculated based on transition frequencies between other levels.
For example the frequency of the 1762 nm transition f(1762) was found
from f(1762)=f(455)-f(614).

References:
[1] - A. Kramida, NIST Atomic Spectra Database (ver. 5.9) (2021)
[2] - A. A. Madej and J. D. Sankey, Phys. Rev. A 41, 2621, (1990)
[3] - N. Yu, W. Nagourney, and H. Dehmelt, Phys. Rev. Lett. 78, 4898 (1997)
[4] - N. J. Stone, Table of nuclear magnetic dipole and electric
  quadrupole moments, Atomic Data and Nuclear Data Tables, Volume 90,
  Issue 1 (2005)
[5] - H. Knab, K. H. Knöll, F. Scheerer and G. Werth, Zeitschrift für
  Physik D Atoms, Molecules and Clusters volume 25, pages205–208 (1993)
[6] - W. Becker, G. Werth, Zeitschrift für Physik A Atoms and Nuclei,
  Volume 311, Issue 1-2, pp. 41-47 (1983)
[7] - P Villemoes et al, J. Phys. B: At. Mol. Opt. Phys. 26 4289 (1993)
[8] - Roger E. Silverans, Gustaaf Borghs, Peter De Bisschop, and
  Marleen Van Hove, Phys. Rev. A 33, 2117 (1986)
[9] - K. Wendt, S. A. Ahmad, F. Buchinger, A. C. Mueller, R. Neugart, and
  E. -W. Otten, Zeitschrift für Physik A Atoms and Nuclei volume 318,
  pages 125–129 (1984)
[10] - Zhiqiang Zhang, K. J. Arnold, S. R. Chanu, R. Kaewuam,
 M. S. Safronova, and M. D. Barrett Phys. Rev. A 101, 062515 (2020)
"""
import numpy as np
import typing
import scipy.constants as consts
import atomic_physics as ap


# level aliases
ground_level = S12 = ap.Level(n=6, S=1 / 2, L=0, J=1 / 2)
P12 = ap.Level(n=6, S=1 / 2, L=1, J=1 / 2)
P32 = ap.Level(n=6, S=1 / 2, L=1, J=3 / 2)
D32 = ap.Level(n=5, S=1 / 2, L=2, J=3 / 2)
shelf = D52 = ap.Level(n=5, S=1 / 2, L=2, J=5 / 2)


class Ba135(ap.Atom):
    def __init__(
        self,
        *,
        B: typing.Optional[float] = None,
        level_filter: typing.Optional[typing.List[ap.Level]] = None
    ):
        """135Ba+ atomic structure.

        :param B: B-field in Tesla (can be changed using :meth setB:)
        :param level_filter: list of Levels to include in the simulation, if
            None we include all levels.
        """
        levels = {
            ground_level: ap.LevelData(
                Ahfs=3591.67011745e6 * consts.h,  # [6]
                g_J=2.0024906,  # [5]
                g_I=(2 / 3) * 0.83794,  # [4]
            ),
            P12: ap.LevelData(
                Ahfs=664.6e6 * consts.h,  # [7]
                g_I=(2 / 3) * 0.83794,  # [4]
            ),
            P32: ap.LevelData(
                Ahfs=113.0e6 * consts.h,  # [7]
                Bhfs=59.0e6 * consts.h,  # [7]
                g_I=(2 / 3) * 0.83794,  # [4]
            ),
            D32: ap.LevelData(
                Ahfs=169.5898e6 * consts.h,  # [8]
                Bhfs=28.9528 * consts.h,  # [8]
                g_I=(2 / 3) * 0.83794,  # [4]
            ),
            D52: ap.LevelData(
                Ahfs=-10.735e6 * consts.h,  # [8]
                Bhfs=38.692e6 * consts.h,  # [8]
                g_I=(2 / 3) * 0.83794,  # [4]
            ),
        }

        transitions = {
            "493": ap.Transition(
                lower=S12,
                upper=P12,
                A=9.53e7,  # [1]
                freq=2 * np.pi * 607426317511042.5,  # [1], [9]
            ),
            "455": ap.Transition(
                lower=S12,
                upper=P32,
                A=1.11e8,  # [1]
                freq=2 * np.pi * 658116515417266.0,
            ),
            "650": ap.Transition(
                lower=D32,
                upper=P12,
                A=3.1e7,  # [1]
                freq=2 * np.pi * 461311910409954.94,  # [1], [7]
            ),
            "585": ap.Transition(
                lower=D32,
                upper=P32,
                A=6.0e6,  # [1]
                freq=2 * np.pi * 512002108316178.56,  # [1], [7]
            ),
            "614": ap.Transition(
                lower=D52,
                upper=P32,
                A=4.12e7,  # [1]
                freq=2 * np.pi * 487990081496443.94,  # [1], [7]
            ),
            "1762": ap.Transition(
                lower=S12,
                upper=D52,
                A=1 / 30.14,  # [10]
                freq=2 * np.pi * 170126433920822.06,
            ),
            "2051": ap.Transition(
                lower=S12,
                upper=D32,
                A=12.5e-3,  # [3]
                freq=2 * np.pi * 146114407101087.56,
            ),
        }

        super().__init__(
            B=B,
            I=3 / 2,
            levels=levels,
            transitions=transitions,
            level_filter=level_filter,
        )
