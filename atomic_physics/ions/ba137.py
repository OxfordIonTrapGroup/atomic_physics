""" 137Ba+

TODO: We need a reference for the Einstein A values
A cursory look turned up:
    Kramida, A., et al., NIST Atomic Spectra Database (ver. 5.8), [Online].
However, this data is based of calculations done in the late 60s. These are
only ~10% accurate.

The transition frequencies are calculated based on the 138Ba+ frequencies
from [1] and the isotope shifts in the second reference listed next to
the frequency. Where no references are given, the transition frequencies
were calculated based on transition frequencies between other levels.
For example the frequency of the 1762 nm transition f(1762) was found
from f(1762)=f(455)-f(614).

References:
[1] - A. Kramida, NIST Atomic Spectra Database (ver. 5.9) (2021)
[2] - A. A. Madej and J. D. Sankey, Phys. Rev. A 41, 2621 (1990)
[3] - N. Yu, W. Nagourney, and H. Dehmelt, Phys. Rev. Lett. 78, 4898 (1997)
[4] - N. J. Stone, Table of nuclear magnetic dipole and electric
  quadrupole moments, Atomic Data and Nuclear Data Tables, Volume 90,
  Issue 1 (2005)
[5] - H. Knab, K. H. Knöll, F. Scheerer and G. Werth, Zeitschrift für
  Physik D Atoms, Molecules and Clusters volume 25, pages 205–208 (1993)
[6] - R. Blatt and G. Werth, Phys. Rev. A 25, 1476 (1982)
[7] - P Villemoes et al, J. Phys. B: At. Mol. Opt. Phys. 26 4289 (1993)
[8] - Nicholas C. Lewty, Boon Leng Chuah, Radu Cazan, B. K. Sahoo, and
  M. D. Barrett, Opt. Express 21, 7131-7132 (2013)
[9] - Nicholas C. Lewty, Boon Leng Chuah, Radu Cazan, Murray D. Barrett,
  and B. K. Sahoo, Phys. Rev. A 88, 012518 (2013)
[10] - K. Wendt, S. A. Ahmad, F. Buchinger, A. C. Mueller, R. Neugart, and
  E. -W. Otten, Zeitschrift für Physik A Atoms and Nuclei volume 318,
  pages 125–129 (1984)
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


class Ba137(ap.Atom):
    def __init__(
        self,
        *,
        B: typing.Optional[float] = None,
        level_filter: typing.Optional[typing.List[ap.Level]] = None
    ):
        """137Ba+ atomic structure.

        :param B: B-field in Tesla (can be changed using :meth setB:)
        :param level_filter: list of Levels to include in the simulation, if
            None we include all levels.
        """
        levels = {
            ground_level: ap.LevelData(
                Ahfs=4018.87083384e6 * consts.h,  # [6]
                g_J=2.0024906,  # [5]
                g_I=(2 / 3) * 0.93737,  # [4]
            ),
            P12: ap.LevelData(
                Ahfs=743.7e6 * consts.h,  # [7]
                g_I=(2 / 3) * 0.93737,  # [4]
            ),
            P32: ap.LevelData(
                Ahfs=127.2e6 * consts.h,  # [7]
                Bhfs=92.5e6 * consts.h,  # [7]
                g_I=(2 / 3) * 0.93737,  # [4]
            ),
            D32: ap.LevelData(
                Ahfs=189.731101e6 * consts.h,  # [8]
                Bhfs=44.536612e6 * consts.h,  # [8]
                g_I=(2 / 3) * 0.93737,  # [4]
            ),
            D52: ap.LevelData(
                Ahfs=-12.029234e6 * consts.h,  # [9]
                Bhfs=59.52552e6 * consts.h,  # [9]
                g_I=(2 / 3) * 0.93737,  # [4]
            ),
        }

        transitions = {
            "493": ap.Transition(
                lower=S12,
                upper=P12,
                A=9.53e7,  # [1]
                freq=2 * np.pi * 607426317510965.0,  # [1], [10]
            ),
            "455": ap.Transition(
                lower=S12,
                upper=P32,
                A=1.11e8,  # [1]
                freq=2 * np.pi * 658116515417166.6,
            ),
            "650": ap.Transition(
                lower=D32,
                upper=P12,
                A=3.1e7,  # [1]
                freq=2 * np.pi * 461311910409885.25,  # [1], [7]
            ),
            "585": ap.Transition(
                lower=D32,
                upper=P32,
                A=6.0e6,  # [1]
                freq=2 * np.pi * 512002108316086.9,  # [1], [7]
            ),
            "614": ap.Transition(
                lower=D52,
                upper=P32,
                A=4.12e7,  # [1]
                freq=2 * np.pi * 487990081496344.9,  # [1], [7]
            ),
            "1762": ap.Transition(
                lower=S12,
                upper=D52,
                A=29e-3,  # [2]
                freq=2 * np.pi * 170126433920821.75,
            ),
            "2051": ap.Transition(
                lower=S12,
                upper=D32,
                A=12.5e-3,  # [3]
                freq=2 * np.pi * 146114407101079.75,
            ),
        }

        super().__init__(
            B=B,
            I=3 / 2,
            levels=levels,
            transitions=transitions,
            level_filter=level_filter,
        )
