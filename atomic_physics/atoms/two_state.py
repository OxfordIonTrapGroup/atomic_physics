r"""Ideal spin 1/2 atom with two states."""

import scipy.constants as consts

from atomic_physics.core import AtomFactory, Level, LevelData


def field_for_frequency(frequency: float) -> float:
    """Returns the B-field needed to produce a given transition frequency.

    :param frequency: the desired transition frequency (rad/s).
    :return: the required magnetic field (T).
    """
    # E = h * f = gJ * uB * B
    uB = consts.physical_constants["Bohr magneton"][0]
    gJ = -consts.physical_constants["electron g factor"][0]
    B = consts.hbar * frequency / (gJ * uB)
    return B


class TwoStateAtomFactory(AtomFactory):
    r""":class:`~atomic_physics.core.AtomFactory` for ideal spin 1/2 atoms.

    Attributes:
        ground_level: the only level within the :class:`TwoStateAtomFactory`.
        upper_state: index of the M=+1/2 (higher-energy) state.
        lower_state: index of the M=-1/2 (lower-energy) state.
    """

    ground_level: Level = Level(n=0, S=1 / 2, L=0, J=1 / 2)
    upper_state = 0
    lower_state = 1

    def __init__(self):
        super().__init__(
            nuclear_spin=0.0,
            level_data=(LevelData(level=self.ground_level, Ahfs=0, Bhfs=0),),
            transitions={},
        )


TwoStateAtom = TwoStateAtomFactory()
"""Ideal spin 1/2 atom with two states."""
