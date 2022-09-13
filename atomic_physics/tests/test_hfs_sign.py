"""Test hfs constant sign"""
import unittest
from atomic_physics.ions import ba133, ba137


class TesLevel(unittest.TestCase):
    """
    Test hfs constant sign
    """

    def test_positive_hfs(self):
        level = ba137.ground_level
        ion = ba137.Ba137(B=100e-4, level_filter=[level])

        # check that states in F=1 have indices 0-2
        for i in range(3):
            self.assertEqual(ion.index(level, -(i-1), F=1), i)

        # check that states in F=2 have indices 3-7
        for i in range(5):
            self.assertEqual(ion.index(level, i-2, F=2), i+3)

    def test_negative_hfs(self):
        level = ba133.ground_level
        ion = ba133.Ba133(B=100e-4, level_filter=[level])

        # check that states in F=1 have indices 0-2
        for i in range(3):
            self.assertEqual(ion.index(level, i-1, F=1), i)

        # check that F=0, mF=0 has index 3
        self.assertEqual(ion.index(level, 0, F=0), 3)

