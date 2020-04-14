import numpy as np
import unittest
from ion_phys.ions.ca43 import Ca43
from ion_phys.rate_equations import Rates
from pprint import pprint

class TestSpont(unittest.TestCase):
    def test_spont(self):
        """ ...
        """
        ion = Ca43(1e-8)
        rates = Rates(ion)
        rates.get_spont()



if __name__ == '__main__':
    unittest.main()
