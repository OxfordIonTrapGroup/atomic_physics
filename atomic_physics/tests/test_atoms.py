import unittest

import numpy as np
from scipy import constants as consts

from atomic_physics.atoms import two_state
from atomic_physics.ions import ba133, ba135, ba137, ba138, ca40, ca43, mg25, sr88


class TestAtoms(unittest.TestCase):
    """Basic smoke testing for all atom/ion class definitions."""

    def test_atoms(self):
        atoms = (
            ba133.Ba133,
            ba135.Ba135,
            ba137.Ba137,
            ba138.Ba138,
            ca40.Ca40,
            ca43.Ca43,
            mg25.Mg25,
            sr88.Sr88,
            two_state.TwoStateAtom,
        )

        # check we can construct the atom without error
        for factory in atoms:
            atom = factory(magnetic_field=1.0)

            # we have a convention of naming atomic transitions by their wavelength
            # in nanometres so check those match the frequency data
            for transition_name, transition in atom.transitions.items():
                np.testing.assert_allclose(
                    transition.frequency / (2 * np.pi),
                    consts.c / (1e-9 * float(transition_name)),
                    rtol=1e-2,
                )

        # check the pre-defined levels exist
        atoms = (
            (ba133, ba133.Ba133),
            (ba135, ba135.Ba135),
            (ba137, ba137.Ba137),
            (ba138, ba138.Ba138),
            (ca40, ca40.Ca40),
            (ca43, ca43.Ca43),
            # (mg25, mg25.Mg25),
            (sr88, sr88.Sr88),
            # (two_state, two_state.TwoStateAtom)
        )
        level_names = ("S12", "P12", "P32", "D32", "D52", "ground_level", "shelf")
        for module, factory in atoms:
            atom = factory(1.0)
            levels = [getattr(module, level_name) for level_name in level_names]
            assert set(atom.levels) == set(levels)

        atoms = ((mg25, mg25.Mg25),)
        level_names = (
            "S12",
            "P12",
            "P32",
            "ground_level",
        )
        for module, factory in atoms:
            atom = factory(1.0)
            levels = [getattr(module, level_name) for level_name in level_names]
            assert set(atom.levels) == set(levels)

        atom = two_state.TwoStateAtom(magnetic_field=1.0)
        assert set(atom.levels) == set((two_state.ground_level,))
        assert two_state.upper_state == 0
        assert two_state.lower_state == 1
