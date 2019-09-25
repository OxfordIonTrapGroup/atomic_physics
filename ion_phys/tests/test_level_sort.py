import unittest
import collections
import numpy as np
import scipy.constants as consts
from ion_phys.utils import sort_levels


def trans_freq(lower, upper):
    E_lower = sum(range(lower+1))/10
    E_upper = sum(range(upper+1))/10
    return (E_upper - E_lower)/consts.h


def add_transition(atom, lower, upper):
    atom["transitions"]["{}_{}".format(lower, upper)] = {
        "lower": lower,
        "upper": upper,
        "f0": trans_freq(lower, upper)
    }


class TestLevelSort(unittest.TestCase):

    def check_order(self, atom):
        sort_levels(atom)
        levels = np.array(list(atom["levels"].keys()))
        levels.sort()
        self.assertEqual(len(atom["sorted_levels"]), len(atom["levels"]))
        for idx, level in enumerate(atom["sorted_levels"]):
            self.assertEqual(level["level"], idx)
            self.assertTrue(abs(level["energy"] - sum(range(idx+1))/10) < 1e-3)

    def test_add_above(self):
        atom = {}
        atom["levels"] = {idx: {} for idx in range(3)}
        atom["transitions"] = collections.OrderedDict()
        add_transition(atom, 0, 1)
        add_transition(atom, 1, 2)
        self.check_order(atom)

    def test_add_below(self):
        atom = {}
        atom["levels"] = {idx: {} for idx in range(3)}
        atom["transitions"] = collections.OrderedDict()
        add_transition(atom, 1, 2)
        add_transition(atom, 0, 1)
        self.check_order(atom)

    def test_add_between_above(self):
        atom = {}
        atom["levels"] = {idx: {} for idx in range(5)}
        atom["transitions"] = collections.OrderedDict()
        add_transition(atom, 0, 4)
        add_transition(atom, 0, 1)
        add_transition(atom, 0, 3)
        add_transition(atom, 0, 2)
        self.check_order(atom)

    def test_add_between_below(self):
        atom = {}
        atom["levels"] = {idx: {} for idx in range(5)}
        atom["transitions"] = collections.OrderedDict()
        add_transition(atom, 3, 4)
        add_transition(atom, 2, 3)
        add_transition(atom, 0, 2)
        add_transition(atom, 1, 2)
        self.check_order(atom)

    def test_double_lambda(self):
        atom = {}
        atom["levels"] = {idx: {} for idx in range(5)}
        atom["transitions"] = collections.OrderedDict()
        add_transition(atom, 1, 3)
        add_transition(atom, 2, 4)
        add_transition(atom, 0, 4)
        add_transition(atom, 0, 3)
        self.check_order(atom)


if __name__ == "__main__":
    unittest.main()
