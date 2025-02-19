import unittest

import numpy as np
import scipy.constants as consts

from atomic_physics.atoms.two_state import TwoStateAtom, field_for_frequency


class TestTwoStateAtom(unittest.TestCase):
    """Test for the two-state atom class."""

    def test_frequency(self):
        w_0 = 2 * np.pi * 100e6
        magnetic_field = field_for_frequency(w_0)
        atom = TwoStateAtom(magnetic_field=magnetic_field)

        np.testing.assert_allclose(
            atom.level_data[TwoStateAtom.ground_level].g_J,
            -consts.physical_constants["electron g factor"][0],
        )
        np.testing.assert_allclose(
            atom.get_transition_frequency_for_states(
                (TwoStateAtom.upper_state, TwoStateAtom.lower_state)
            ),
            w_0,
        )
