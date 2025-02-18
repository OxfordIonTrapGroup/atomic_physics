import unittest

import numpy as np
from scipy import constants as consts

from atomic_physics.atoms.two_state import TwoStateAtom
from atomic_physics.ions.ba133 import Ba133
from atomic_physics.ions.ba135 import Ba135
from atomic_physics.ions.ba137 import Ba137
from atomic_physics.ions.ba138 import Ba138
from atomic_physics.ions.ca40 import Ca40
from atomic_physics.ions.ca43 import Ca43
from atomic_physics.ions.mg25 import Mg25
from atomic_physics.ions.sr88 import Sr88


class TestAtoms(unittest.TestCase):
    """Basic smoke testing for all atom/ion class definitions."""

    def test_atoms(self):
        atoms = (
            Ba133,
            Ba135,
            Ba137,
            Ba138,
            Ca40,
            Ca43,
            Mg25,
            Sr88,
            TwoStateAtom,
        )

        # check we can construct the atom without error
        for factory in atoms:
            atom = factory(magnetic_field=1.0)

            # we have a convention of naming atomic transitions by their wavelength
            # in nanometres so check those match the frequency data
            for transition_name, transition in atom.transitions.items():
                np.testing.assert_allclose(
                    consts.c / (transition.frequency / (2 * np.pi)) * 1e9,
                    float(transition_name),
                    atol=1,
                )

        # check the pre-defined levels exist
        atoms_d = (Ba133, Ba135, Ba137, Ba138, Ca40, Ca43, Sr88)

        level_names = ("S12", "P12", "P32", "D32", "D52", "ground_level", "shelf")
        for factory in atoms_d:
            atom = factory(1.0)
            levels = [getattr(factory, level_name) for level_name in level_names]
            assert set(atom.levels) == set(levels)

        atoms_no_d = (Mg25,)
        level_names = (
            "S12",
            "P12",
            "P32",
            "ground_level",
        )
        for factory in atoms_no_d:
            atom = factory(1.0)
            levels = [getattr(factory, level_name) for level_name in level_names]
            assert set(atom.levels) == set(levels)

        atom = TwoStateAtom(magnetic_field=1.0)
        assert set(atom.levels) == set((TwoStateAtom.ground_level,))
        assert TwoStateAtom.upper_state == 0
        assert TwoStateAtom.lower_state == 1
