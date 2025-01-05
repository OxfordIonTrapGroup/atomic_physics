import unittest

import numpy as np
import scipy.constants as consts

from atomic_physics.atoms import two_state


class TestTwoStateAtom(unittest.TestCase):
    """Test for the two-state atom class."""

    def test_frequency(self):
        w_0 = 2 * np.pi * 100e6
        magnetic_field = two_state.field_for_frequency(w_0)
        atom = two_state.TwoStateAtom(magnetic_field=magnetic_field)

        np.testing.assert_allclose(
            atom.level_data[two_state.ground_level].g_J,
            -consts.physical_constants["electron g factor"][0],
        )
        np.testing.assert_allclose(
            atom.get_transition_frequency_for_states(
                (two_state.upper_state, two_state.lower_state)
            ),
            w_0,
        )
