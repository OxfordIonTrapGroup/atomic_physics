import unittest

from atomic_physics.core import Laser
from atomic_physics.ions import ca43
from atomic_physics.rate_equations import Rates


class TestStim(unittest.TestCase):
    def test_multi_transition(self):
        """Test with lasers on multiple transitions (see #15)"""
        ion = ca43.Ca43(magnetic_field=146e-4)
        rates = Rates(ion)
        Lasers = (
            Laser("397", polarization=0, intensity=1, detuning=0),
            Laser("866", polarization=0, intensity=1, detuning=0),
        )
        rates.get_transitions_matrix(Lasers)

    def test_multi_laser(self):
        """Test with multiple lasers on one transition"""
        ion = ca43.Ca43(magnetic_field=146e-4)
        rates = Rates(ion)
        Lasers = (
            Laser("397", polarization=0, intensity=1, detuning=0),
            Laser("397", polarization=+1, intensity=1, detuning=0),
        )
        rates.get_transitions_matrix(Lasers)
