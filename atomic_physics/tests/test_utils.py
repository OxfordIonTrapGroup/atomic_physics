import unittest

import scipy.constants as consts
import numpy as np
from atomic_physics.ions import ca40, ca43, ba137
from atomic_physics.utils import (
    ac_zeeman_shift_for_transition,
    ac_zeeman_shift_for_state,
)
from atomic_physics.core import RFDrive
from atomic_physics.polarization import (
    PI_POLARIZATION,
    SIGMA_MINUS_POLARIZATION,
    SIGMA_PLUS_POLARIZATION,
)


def ac_zeeman(Omega, w_transition, w_rf):
    return 0.5 * (Omega**2) * (w_transition / (w_transition**2 - w_rf**2))


class TestACZeeman(unittest.TestCase):
    """Tests for `atomic_physics.utils`.

    References:
        [1] Thomas Harty DPhil Thesis (2013).
    """

    def test_acz_ca40(self):
        """
        Test the AC Zeeman shift for the ground level qubit in a 40Ca+ ion.
        """
        B_lab = 10e-4  # quantization field
        B_rf = 1e-6  # RF field of 1 uT
        level = ca40.ground_level
        Ca40 = ca40.Ca40.filter_levels(level_filter=(level,))
        ion = Ca40(magnetic_field=B_lab)

        f = abs(ion.get_transition_frequency_for_states((0, 1), relative=True))
        mu = ion.get_magnetic_dipoles()[0, 1]
        rabi_freq = mu * B_rf / consts.hbar
        f_mode = 2 * np.pi * 3e6  # assuming 3 MHz motional mode frequency
        f_rf = f + f_mode

        # Calculate the absolute value of the expected AC Zeeman shift for each state.
        expected_shift_abs = np.abs(ac_zeeman(rabi_freq, f, f_rf))

        rf_drive_plus = RFDrive(
            frequency=f_rf, amplitude=B_rf, polarization=SIGMA_PLUS_POLARIZATION
        )
        rf_drive_minus = RFDrive(
            frequency=f_rf, amplitude=B_rf, polarization=SIGMA_MINUS_POLARIZATION
        )
        rf_drive_pi = RFDrive(
            frequency=f_rf, amplitude=B_rf, polarization=PI_POLARIZATION
        )

        # All pi shifts should be zero since there are no pi spectator transitions.
        acz_pi_0 = ac_zeeman_shift_for_state(ion, 0, rf_drive_pi)
        acz_pi_1 = ac_zeeman_shift_for_state(ion, 1, rf_drive_pi)
        acz_pi_transition = ac_zeeman_shift_for_transition(ion, [0, 1], rf_drive_pi)
        self.assertEqual([acz_pi_0, acz_pi_1, acz_pi_transition], [0.0, 0.0, 0.0])

        # All sigma minus shifts should be zero since there are no sigma minus
        # spectator transitions.
        acz_sigma_m_0 = ac_zeeman_shift_for_state(ion, 0, rf_drive_minus)
        acz_sigma_m_1 = ac_zeeman_shift_for_state(ion, 1, rf_drive_minus)
        acz_sigma_m_transition = ac_zeeman_shift_for_transition(
            ion, [0, 1], rf_drive_minus
        )
        self.assertEqual(
            [acz_sigma_m_0, acz_sigma_m_1, acz_sigma_m_transition], [0.0, 0.0, 0.0]
        )

        acz_sigma_p_0 = ac_zeeman_shift_for_state(ion, 0, rf_drive_plus)
        acz_sigma_p_1 = ac_zeeman_shift_for_state(ion, 1, rf_drive_plus)
        acz_sigma_p_transition = ac_zeeman_shift_for_transition(
            ion, [0, 1], rf_drive_plus
        )
        # The higher energy state moves down in energy by the expected shift and
        # the lower energy state moves up in energy by the expected shift since
        # the RF frequency is higher than the transition frequency. The frequency
        # of the transition decreases by double the expected shift.
        self.assertAlmostEqual(acz_sigma_p_0, -expected_shift_abs, delta=1e-8)
        self.assertAlmostEqual(acz_sigma_p_1, expected_shift_abs, delta=1e-8)
        self.assertAlmostEqual(
            acz_sigma_p_transition, -2 * expected_shift_abs, delta=1e-8
        )

    def test_acz_sideband_ca43(self):
        """
        Test the AC Zeeman shift for ground level qubits due to motional mode
        sidebands in a 43Ca+ ion and compare to Chapter 6 in [1].
        """
        B_lab = 146e-4  # quantization field
        B_rf = 1e-6  # RF field of 1 uT
        level = ca43.ground_level
        Ca43 = ca43.Ca43.filter_levels(level_filter=(level,))
        ion = Ca43(magnetic_field=B_lab)

        # RF frequency calculated relative to the transition frequency of the
        # (4, 0) <-> (3, 1) qubit pair.
        f = abs(
            ion.get_transition_frequency_for_states(
                (
                    ion.get_state_for_F(level, F=4, M_F=0),
                    ion.get_state_for_F(level, F=3, M_F=+1),
                ),
                relative=True,
            )
        )

        f_mode = 2 * np.pi * 3e6  # assuming 3 MHz motional mode frequency

        # Qubit pairs in Tables 6.3, 6.4, and 6.5 in [1] and the expected shifts
        # in order [bsb_pi, rsb_pi, bsb_sigma, rsb_sigma], respectively.
        qubit_pairs_and_shifts = [
            ((4, 0), (3, 0), 2 * np.pi * np.array([2.005, 1.786, -15.344, 16.836])),
            ((4, 0), (3, 1), 2 * np.pi * np.array([0.126, -0.0926, -16.298, 15.864])),
            ((4, 1), (3, 1), 2 * np.pi * np.array([-1.753, -1.971, -16.341, 15.840])),
        ]

        rf_bsb_pi = RFDrive(
            frequency=f + f_mode, amplitude=B_rf, polarization=PI_POLARIZATION
        )

        rf_rsb_pi = RFDrive(
            frequency=f - f_mode, amplitude=B_rf, polarization=PI_POLARIZATION
        )

        rf_bsb_sigma = RFDrive(
            frequency=f + f_mode,
            amplitude=B_rf,
            polarization=SIGMA_MINUS_POLARIZATION + SIGMA_PLUS_POLARIZATION,
        )

        rf_rsb_sigma = RFDrive(
            frequency=f - f_mode,
            amplitude=B_rf,
            polarization=SIGMA_MINUS_POLARIZATION + SIGMA_PLUS_POLARIZATION,
        )

        for qubit in qubit_pairs_and_shifts:
            idx0 = ion.get_state_for_F(level, qubit[0][0], qubit[0][1])
            idx1 = ion.get_state_for_F(level, qubit[1][0], qubit[1][1])

            acz_diff_bsb_pi = ac_zeeman_shift_for_transition(
                ion, [idx0, idx1], rf_bsb_pi
            )

            acz_diff_rsb_pi = ac_zeeman_shift_for_transition(
                ion, [idx0, idx1], rf_rsb_pi
            )

            acz_diff_bsb_sigma = ac_zeeman_shift_for_transition(
                ion, [idx0, idx1], rf_bsb_sigma
            )

            acz_diff_rsb_sigma = ac_zeeman_shift_for_transition(
                ion, [idx0, idx1], rf_rsb_sigma
            )

            np.testing.assert_allclose(
                [
                    acz_diff_bsb_pi,
                    acz_diff_rsb_pi,
                    acz_diff_bsb_sigma,
                    acz_diff_rsb_sigma,
                ],
                qubit[2],
                rtol=1e-3,
            )

    def test_acz_trap_rf_ca43(self):
        """
        Test the AC Zeeman shift for the (4, 0) <-> (3, 1) qubit pair in 43Ca+ at
        146 G due to a 38.2 MHz trap RF and compare to Table 5.9 in [1].
        """
        B_lab = 146e-4  # quantization field
        B_rf = 1e-6  # RF field of 1 uT
        level = ca43.ground_level
        Ca43 = ca43.Ca43.filter_levels(level_filter=(level,))
        ion = Ca43(magnetic_field=B_lab)
        idx_qubit_0 = ion.get_state_for_F(level, F=4, M_F=0)
        idx_qubit_1 = ion.get_state_for_F(level, F=3, M_F=+1)

        f_rf = 2 * np.pi * 38.2e6  # Trap RF frequency 38.2 MHz
        rf_pi = RFDrive(frequency=f_rf, amplitude=B_rf, polarization=PI_POLARIZATION)

        rf_sigma = RFDrive(
            frequency=f_rf,
            amplitude=B_rf,
            polarization=SIGMA_MINUS_POLARIZATION + SIGMA_PLUS_POLARIZATION,
        )

        acz_diff_rf_pi = ac_zeeman_shift_for_transition(
            ion, [idx_qubit_0, idx_qubit_1], rf_pi
        )

        acz_diff_rf_sigma = ac_zeeman_shift_for_transition(
            ion, [idx_qubit_0, idx_qubit_1], rf_sigma
        )

        np.testing.assert_allclose(
            [acz_diff_rf_pi, acz_diff_rf_sigma],
            2 * np.pi * np.array([0.0604, -0.3976]),
            rtol=5e-3,
        )
