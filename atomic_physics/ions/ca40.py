""" 40Ca+

References:
[1] - A. Kramida, At. Data Nucl. Data Tables 133-134, 101322 (2020)
[2] - T. P. Harty, DPhil Thesis (2013)
[3] - M. Chwalla et all, PRL 102, 023002 (2009)
"""
import numpy as np
import typing
import atomic_physics as ap


# level aliases
ground_level = S12 = ap.Level(n=4, S=1 / 2, L=0, J=1 / 2)
P12 = ap.Level(n=4, S=1 / 2, L=1, J=1 / 2)
P32 = ap.Level(n=4, S=1 / 2, L=1, J=3 / 2)
D32 = ap.Level(n=4, S=1 / 2, L=2, J=3 / 2)
shelf = D52 = ap.Level(n=4, S=1 / 2, L=2, J=5 / 2)


class Ca40(ap.Atom):
    def __init__(
        self,
        *,
        B: typing.Optional[float] = None,
        level_filter: typing.Optional[typing.List[ap.Level]] = None
    ):
        """40Ca+ atomic structure.

        :param B: B-field in Tesla (can be changed using :meth:`setB`)
        :param level_filter: list of Levels to include in the simulation, if
            None we include all levels.
        """
        levels = {
            ground_level: ap.LevelData(g_J=2.00225664),  # [2]
            P12: ap.LevelData(),
            P32: ap.LevelData(),
            D32: ap.LevelData(),
            D52: ap.LevelData(g_J=1.2003340),  # [3]
        }

        transitions = {
            "397": ap.Transition(
                lower=S12,
                upper=P12,
                A=132e6,  # [?]
                freq=2 * np.pi * 755222765771e3,  # [1]
            ),
            "393": ap.Transition(
                lower=S12,
                upper=P32,
                A=135e6,  # [?]
                freq=2 * np.pi * 761905012599e3,  # [1]
            ),
            "866": ap.Transition(
                lower=D32,
                upper=P12,
                A=8.4e6,  # [?]
                freq=2 * np.pi * 346000235016e3,  # [1]
            ),
            "850": ap.Transition(
                lower=D32,
                upper=P32,
                A=0.955e6,  # [?]
                freq=2 * np.pi * 352682481844e3,  # [1]
            ),
            "854": ap.Transition(
                lower=D52,
                upper=P32,
                A=8.48e6,  # [?]
                freq=2 * np.pi * 350862882823e3,  # [1]
            ),
            "729": ap.Transition(
                lower=S12,
                upper=D52,
                A=0.856,  # [?]
                freq=411042129776.4017e3 * 2 * np.pi,  # [1]
            ),
            "733": ap.Transition(
                lower=S12,
                upper=D32,
                A=0.850,  # [?]
                freq=409222530754.868e3 * 2 * np.pi,  # [1]
            ),
        }

        super().__init__(
            B=B, I=0, levels=levels, transitions=transitions, level_filter=level_filter
        )
