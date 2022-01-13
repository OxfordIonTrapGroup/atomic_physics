from ion_phys.ions.ca43 import Ca43
from ion_phys import Laser
from ion_phys.rate_equations import Rates
import unittest


class TestStim(unittest.TestCase):
    def test_multi_transition(self):
        """Test with lasers on multiple transitions (see #15)"""
        ion = Ca43(B=146e-4)
        rates = Rates(ion)
        Lasers = [
            Laser("397", q=0, I=1, delta=0),
            Laser("866", q=0, I=1, delta=0),
        ]
        rates.get_transitions(Lasers)

    def test_multi_laser(self):
        """Test with multiple lasers on one transition"""
        ion = Ca43(B=146e-4)
        rates = Rates(ion)
        Lasers = [
            Laser("397", q=0, I=1, delta=0),
            Laser("397", q=+1, I=1, delta=0),
        ]
        rates.get_transitions(Lasers)
