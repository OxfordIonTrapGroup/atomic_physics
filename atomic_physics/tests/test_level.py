"""Test Spin 1/2 nuclei"""
import unittest
from atomic_physics.ions import ca40


class TesLevel(unittest.TestCase):
    """
    Test level assignment
    """

    def test_level(self):
        ion = ca40.Ca40(B=100e-4, level_filter=[ca40.ground_level, ca40.shelf])

        # check that states with indices 0 and 1 belong to S1/2
        for i in range(2):
            self.assertEqual(ion.level(i).L, 0)
            self.assertEqual(ion.level(i).J, 0.5)

        # check that states with indices 2-7 belong to D5/2
        for i in range(6):
            self.assertEqual(ion.level(i + 2).L, 2)
            self.assertEqual(ion.level(i + 2).J, 2.5)
