"""Test Rates Calculations"""

import unittest

import numpy as np

from atomic_physics.core import Laser
from atomic_physics.ions import ca43
from atomic_physics.rate_equations import Rates


def _steady_state_population(intensity: float):
    "Steady state population in the P-state for resonant intensity /I0"
    return intensity / (2 * intensity + 1)


class TestTSS(unittest.TestCase):
    """Two-state system tests.

    The closed stretch cycling transition is used to make a two state system.
    """

    def test_rates_relations(self):
        """Test the spontaneous rates satisfy relations in net rates

        This relation is used in the steady states tests.
        """
        intensity_list = [1e-3, 1e-1, 0.3, 1, 1.0, 2, 10.0, 1.2e4]

        Ca43 = ca43.Ca43.filter_levels(level_filter=(ca43.ground_level, ca43.P32))
        ion = Ca43(magnetic_field=5e-4)
        s_idx = ion.get_state_for_F(ca43.ground_level, F=4, M_F=+4)
        p_idx = ion.get_state_for_F(ca43.P32, F=5, M_F=+5)

        rates = Rates(ion)
        detuning = ion.get_transition_frequency_for_states((s_idx, p_idx))
        for intensity in intensity_list:
            Lasers = (
                Laser("393", polarization=+1, intensity=intensity, detuning=detuning),
            )  # resonant
            trans = rates.get_transitions_matrix(Lasers)

            spont = rates.get_spont_matrix()
            r = spont[p_idx, p_idx] / (trans[p_idx, p_idx] + trans[p_idx, s_idx])
            self.assertAlmostEqual(r, 1.0, places=7)

    def test_steady_state(self):
        """Check that the steady-state solution is found correctly."""
        ion = ca43.Ca43(magnetic_field=5e-4)
        rates = Rates(ion)
        lasers = (
            Laser("397", polarization=+1, intensity=1, detuning=0),
            Laser("397", polarization=-1, intensity=1, detuning=0),
            Laser("866", polarization=+1, intensity=1, detuning=0),
            Laser("866", polarization=-1, intensity=1, detuning=0),
        )
        transitions = rates.get_transitions_matrix(lasers)
        steady_state = rates.get_steady_state_populations(transitions)
        np.testing.assert_allclose(transitions @ steady_state, 0, atol=1e-8)

    def test_steady_state_intensity(self):
        """Test the steady state intensity scaling"""

        # use both integers and floats
        intensity_list = [1e-3, 1e-1, 0.3, 1, 1.0, 2, 10.0, 1.2e4]

        Ca43 = ca43.Ca43.filter_levels(level_filter=(ca43.ground_level, ca43.P32))
        ion = Ca43(magnetic_field=5e-4)

        s_idx = ion.get_state_for_F(ca43.ground_level, F=4, M_F=+4)
        p_idx = ion.get_state_for_F(ca43.P32, F=5, M_F=+5)

        rates = Rates(ion)
        detuning = ion.get_transition_frequency_for_states((s_idx, p_idx))

        for intensity in intensity_list:
            Lasers = (
                Laser("393", polarization=+1, intensity=intensity, detuning=detuning),
            )  # resonant
            trans = rates.get_transitions_matrix(Lasers)

            Np_ss = _steady_state_population(intensity)
            # transition rates normalised by A coefficient
            dNp_dt = trans[p_idx, p_idx] * Np_ss + trans[p_idx, s_idx] * (1 - Np_ss)
            dNp_dt = dNp_dt / (trans[p_idx, p_idx] + trans[p_idx, s_idx])
            self.assertAlmostEqual(0.0, dNp_dt, places=7)
            dNs_dt = trans[s_idx, p_idx] * Np_ss + trans[s_idx, s_idx] * (1 - Np_ss)
            dNs_dt = dNs_dt / (trans[s_idx, p_idx] + trans[s_idx, s_idx])
            self.assertAlmostEqual(0.0, dNs_dt, places=7)

    def test_steady_state_detuning(self):
        """Test steady state detuning dependence"""

        # assume 1 saturation intensity
        Ca43 = ca43.Ca43.filter_levels(level_filter=(ca43.ground_level, ca43.P32))
        ion = Ca43(magnetic_field=5e-4)

        s_idx = ion.get_state_for_F(ca43.ground_level, F=4, M_F=+4)
        p_idx = ion.get_state_for_F(ca43.P32, F=5, M_F=+5)

        rates = Rates(ion)
        detuning = ion.get_transition_frequency_for_states((s_idx, p_idx))

        Lasers = (
            Laser("393", polarization=+1, intensity=1.0, detuning=detuning),
        )  # resonant
        trans = rates.get_transitions_matrix(Lasers)
        line_width = abs(trans[p_idx, p_idx] + trans[p_idx, s_idx])

        # detuning scan relative to linewidth
        norm_detuning = [-1e4, 2.3e1, 2, -4, 0.5, 0]
        for det in norm_detuning:
            I_eff = 1 / (4 * det**2 + 1)
            Np_ss = _steady_state_population(I_eff)

            Lasers = (
                Laser(
                    "393",
                    polarization=+1,
                    intensity=1.0,
                    detuning=detuning + line_width * det,
                ),
            )
            trans = rates.get_transitions_matrix(Lasers)

            # transition rates normalised by A coefficient
            dNp_dt = trans[p_idx, p_idx] * Np_ss + trans[p_idx, s_idx] * (1 - Np_ss)
            dNp_dt = dNp_dt / (trans[p_idx, p_idx] + trans[p_idx, s_idx])
            self.assertAlmostEqual(0.0, dNp_dt, places=7)
            dNs_dt = trans[s_idx, p_idx] * Np_ss + trans[s_idx, s_idx] * (1 - Np_ss)
            dNs_dt = dNs_dt / (trans[s_idx, p_idx] + trans[s_idx, s_idx])
            self.assertAlmostEqual(0.0, dNs_dt, places=7)


if __name__ == "__main__":
    unittest.main()
