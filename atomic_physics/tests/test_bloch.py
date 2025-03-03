import unittest

import numpy as np

from atomic_physics.atoms.two_state import TwoStateAtom, field_for_frequency
from atomic_physics.bloch import HAS_JULIA, Bloch
from atomic_physics.core import RFDrive
from atomic_physics.polarization import SIGMA_PLUS_POLARIZATION


class TestBloch(unittest.TestCase):
    """Tests for optical Bloch equations solver."""

    def setUp(self):
        if not HAS_JULIA:
            self.skipTest("JuliaCall not installed")

    def test_rf_two_state(self):
        """Test driving a two-state system with an RF field."""

        rabi = 2 * np.pi * 1e6
        delta = rabi / 2

        f0 = 100e6 * 2 * np.pi
        b = field_for_frequency(f0)
        atom = TwoStateAtom(b)

        bloch = Bloch(atom)
        amplitude = atom.get_amplitude_for_rabi_rf(
            states=(TwoStateAtom.lower_state, TwoStateAtom.upper_state), rabi=rabi
        )

        H = bloch.hamiltonian_for_rf_drive(
            level=TwoStateAtom.ground_level,
            drive=RFDrive(
                frequency=f0 - delta,
                amplitude=amplitude,
                polarization=SIGMA_PLUS_POLARIZATION,
            ),
        )

        t_pi = np.pi / rabi
        t = np.linspace(0, 3 * t_pi, 1000)
        results = bloch.solve_schroedinger(H, np.array([1, 0]), t)
        P_julia = np.abs(results[:, 0]) ** 2

        omega_eff = np.sqrt(rabi**2 + delta**2)
        P_analytic = 1 - (rabi / omega_eff * np.sin(0.5 * omega_eff * t)) ** 2

        np.testing.assert_allclose(P_julia, P_analytic, atol=5e-6, rtol=5e-6)
