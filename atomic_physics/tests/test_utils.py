import unittest

import numpy as np
import scipy.constants as consts

from atomic_physics.core import RFDrive
from atomic_physics.ions.ca40 import Ca40
from atomic_physics.ions.ca43 import Ca43
from atomic_physics.polarization import (
    PI_POLARIZATION,
    SIGMA_MINUS_POLARIZATION,
    SIGMA_PLUS_POLARIZATION,
)
from atomic_physics.utils import (
    ac_zeeman_shift_for_state,
    ac_zeeman_shift_for_transition,
    d2omega_dB2,
    domega_dB,
    field_insensitive_point,
    rayleigh_range,
)


def ac_zeeman(Omega, w_transition, w_rf):
    return 0.5 * (Omega**2) * (w_transition / (w_transition**2 - w_rf**2))


class TestUtils(unittest.TestCase):
    """Tests for ``atomic_physics.utils``"""

    def test_rayleigh_range(self):
        waist = 33e-6
        transition = Ca43.transitions["397"]
        wavelength = consts.c / (transition.frequency / (2 * np.pi))

        np.testing.assert_allclose(
            np.pi * waist**2 / wavelength,
            rayleigh_range(Ca43.transitions["397"], waist),
        )


class TestFieldSensitivity(unittest.TestCase):
    """Tests for field sensitivity helpers.

    References:
        [1] T Harty DPhil Thesis.
    """

    def test_domega_db(self):
        """Tests for ``utils.domega_dB``."""
        ion = Ca43(magnetic_field=146.0942e-4)

        # [1] table E.4
        values = [
            (-4, -3, 2.519894),
            (-3, -3, 2.237093),
            (-3, -2, 1.940390),
            (-2, -3, 1.939817),
            (-2, -2, 1.643113),
            (-2, -1, 1.330090),
            (-1, -2, 1.329517),
            (-1, -1, 1.016493),
            (-1, 0, 0.684932),
            (0, -1, 0.684359),
            (0, 0, 0.352797),
            (0, +1, 0.0),
            (+1, 0, -0.000573),
            (+1, +1, -0.353370),
            (+1, +2, -0.730728),
            (+2, +1, -0.731301),
            (+2, +2, -1.108659),
            (+2, +3, -1.514736),
            (+3, +2, -1.515309),
            (+3, +3, -1.921385),
            (+4, +3, -2.362039),
        ]

        for M4, M3, df_dB_ref in values:
            l_index = ion.get_state_for_F(level=Ca43.ground_level, F=4, M_F=M4)
            u_index = ion.get_state_for_F(level=Ca43.ground_level, F=3, M_F=M3)
            np.testing.assert_allclose(
                domega_dB(
                    atom_factory=Ca43,
                    states=(l_index, u_index),
                    magnetic_field=146.0942e-4,
                )
                / (2 * np.pi * 1e10),
                df_dB_ref,
                atol=1e-6,
            )

    def test_d2omega_db2(self):
        """Tests for ``utils.d2omega_dB2``."""
        ion = Ca43(magnetic_field=146.0942e-4)

        np.testing.assert_allclose(
            d2omega_dB2(
                atom_factory=Ca43,
                magnetic_field=146.0942e-4,
                states=(
                    ion.get_state_for_F(Ca43.S12, F=4, M_F=0),
                    ion.get_state_for_F(Ca43.S12, F=3, M_F=+1),
                ),
            )
            / (2 * np.pi * 1e11),
            2.416,
            atol=1e-3,
        )

    def test_field_insensitive_point(self):
        """Tests for ``utils.field_insensitive_point``."""
        np.testing.assert_allclose(
            field_insensitive_point(
                atom_factory=Ca43,
                level_0=Ca43.S12,
                F_0=4,
                M_F_0=0,
                level_1=Ca43.S12,
                F_1=3,
                M_F_1=+1,
                magnetic_field_guess=10e-4,
            ),
            146.0942e-4,
            atol=1e-4,
        )


class TestACZeeman(unittest.TestCase):
    """Tests for `atomic_physics.utils`.

    References:
        [1] Thomas Harty DPhil Thesis (2013).
    """

    def test_acz_ca40(self):
        """
        Test the AC Zeeman shift for the ground level qubit in a 40Ca+ ion.
        """
        B_rf = 1e-6  # RF field of 1 uT
        ion = Ca40.filter_levels(level_filter=(Ca40.ground_level,))(
            magnetic_field=10e-4
        )

        w_0 = ion.get_transition_frequency_for_states((0, 1))
        mu = ion.get_magnetic_dipoles()[0, 1]
        rabi_freq = mu * B_rf / consts.hbar
        detuning = 2 * np.pi * 3e6
        w_rf = w_0 + detuning  # blue detuning so expect a negative overall shift

        # All pi shifts should be zero since there are no pi spectator transitions.
        rf_drive_pi = RFDrive(
            frequency=w_rf, amplitude=B_rf, polarization=PI_POLARIZATION
        )
        np.testing.assert_allclose(
            (
                ac_zeeman_shift_for_state(ion, 0, rf_drive_pi),
                ac_zeeman_shift_for_state(ion, 1, rf_drive_pi),
                ac_zeeman_shift_for_transition(ion, (0, 1), rf_drive_pi),
            ),
            0.0,
        )

        # All sigma minus shifts should be zero since there are no sigma minus
        # spectator transitions.
        rf_drive_minus = RFDrive(
            frequency=w_rf, amplitude=B_rf, polarization=SIGMA_MINUS_POLARIZATION
        )
        np.testing.assert_allclose(
            (
                ac_zeeman_shift_for_state(ion, 0, rf_drive_minus),
                ac_zeeman_shift_for_state(ion, 1, rf_drive_minus),
                ac_zeeman_shift_for_transition(ion, (0, 1), rf_drive_minus),
            ),
            0.0,
        )

        # Calculate the absolute value of the expected AC Zeeman shift for each state.
        expected_shift_abs = np.abs(ac_zeeman(rabi_freq, w_0, w_rf))

        rf_drive_plus = RFDrive(
            frequency=w_rf, amplitude=B_rf, polarization=SIGMA_PLUS_POLARIZATION
        )
        acz_sigma_p_0 = ac_zeeman_shift_for_state(ion, 0, rf_drive_plus)
        acz_sigma_p_1 = ac_zeeman_shift_for_state(ion, 1, rf_drive_plus)
        acz_sigma_p_transition = ac_zeeman_shift_for_transition(
            ion, (0, 1), rf_drive_plus
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

    def test_acz_ca43(self):
        """Compare AC Zeeman shifts in the ground-level of 43Ca+ at 146G to values from
        [1] Chapter 6.
        """
        B_rf = 1e-6  # RF field of 1 uT
        level = Ca43.ground_level
        factory = Ca43.filter_levels(level_filter=(level,))
        ion = factory(magnetic_field=146.0942e-4)

        # RF frequency calculated relative to the transition frequency of the
        # (4, 0) <-> (3, 1) qubit pair.
        w_0 = ion.get_transition_frequency_for_states(
            (
                ion.get_state_for_F(level, F=4, M_F=0),
                ion.get_state_for_F(level, F=3, M_F=+1),
            ),
        )

        detuning = 2 * np.pi * 3e6

        # Qubit pairs in Tables 6.3, 6.4, and 6.5 in [1] and the expected shifts
        # in order [pi_p, pi_m, sigma_p, sigma_m], where _p and _m denote positive or
        # negative detunings.
        refs = [
            ((4, 0), (3, 0), 2 * np.pi * np.array([2.005, 1.786, -15.344, 16.836])),
            ((4, 0), (3, 1), 2 * np.pi * np.array([0.126, -0.0926, -16.298, 15.864])),
            ((4, 1), (3, 1), 2 * np.pi * np.array([-1.753, -1.971, -16.341, 15.840])),
        ]

        rf_drives = (
            RFDrive(
                frequency=w_0 + detuning, amplitude=B_rf, polarization=PI_POLARIZATION
            ),
            RFDrive(
                frequency=w_0 - detuning, amplitude=B_rf, polarization=PI_POLARIZATION
            ),
            RFDrive(
                frequency=w_0 + detuning,
                amplitude=B_rf,
                polarization=SIGMA_MINUS_POLARIZATION + SIGMA_PLUS_POLARIZATION,
            ),
            RFDrive(
                frequency=w_0 - detuning,
                amplitude=B_rf,
                polarization=SIGMA_MINUS_POLARIZATION + SIGMA_PLUS_POLARIZATION,
            ),
        )

        for (_, M4), (_, M3), ref_shifts in refs:
            idx0 = ion.get_state_for_F(level=level, F=4, M_F=M4)
            idx1 = ion.get_state_for_F(level=level, F=3, M_F=M3)

            shifts = np.array(
                [
                    ac_zeeman_shift_for_transition(ion, (idx0, idx1), rf_drive)
                    for rf_drive in rf_drives
                ]
            )
            np.testing.assert_allclose(
                shifts / (2 * np.pi), ref_shifts / (2 * np.pi), atol=1e-3
            )

    def test_acz_ca43_rf(self):
        """Compare AC Zeeman shifts in the ground-level of 43Ca+ at 146G to values from
        [1] table 5.9.
        """
        level = Ca43.ground_level
        ion = Ca43.filter_levels(level_filter=(level,))(magnetic_field=146.0942e-4)
        idx_qubit_0 = ion.get_state_for_F(level, F=4, M_F=0)
        idx_qubit_1 = ion.get_state_for_F(level, F=3, M_F=+1)

        B_rf = 1e-6
        w_rf = 2 * np.pi * 38.2e6

        drives = (
            RFDrive(frequency=w_rf, amplitude=B_rf, polarization=PI_POLARIZATION),
            RFDrive(
                frequency=w_rf,
                amplitude=B_rf,
                polarization=SIGMA_MINUS_POLARIZATION + SIGMA_PLUS_POLARIZATION,
            ),
        )

        shifts = np.array(
            [
                ac_zeeman_shift_for_transition(ion, (idx_qubit_0, idx_qubit_1), drive)
                for drive in drives
            ]
        )
        np.testing.assert_allclose(
            shifts / (2 * np.pi),
            np.array([0.0604, -0.3976]),
            atol=1e-4,
        )

    def test_acz_ca43_d(self):
        """Test that we calculate the AC Zeeman shifts correct in the D-levels of
        43Ca+.

        We pick a level with J>1/2 and work at intermediate field (lots of state
        mixing) to ensure there is lots going on in these calculations to provide
        a good stress test of the code.
        """
        level = Ca43.D52
        ion = Ca43(magnetic_field=10e-4)

        upper = ion.get_state_for_F(level, F=3, M_F=1)  # state 65
        lower = ion.get_state_for_F(level, F=4, M_F=+1)  # state 70

        # calculate the shift due to pi-polarized radiation
        w_0 = ion.get_transition_frequency_for_states((lower, upper))
        detuning = 2e6 * 2 * np.pi
        w_rf = w_0 + detuning
        drive = RFDrive(frequency=w_rf, amplitude=1e-6, polarization=PI_POLARIZATION)

        # shifts on lower-energy state
        victim = lower
        shift_lower = 0.0

        spectator = ion.get_state_for_F(level, F=1, M_F=1)  # state 50
        w_transition = ion.get_transition_frequency_for_states((victim, spectator))
        Omega = ion.get_rabi_rf(victim, spectator, 1e-6)
        sign = -1  # victim is the lower-energy state in this transition
        shift_lower += sign * ac_zeeman(Omega, w_transition, w_rf)

        spectator = ion.get_state_for_F(level, F=2, M_F=1)  # state 57
        w_transition = ion.get_transition_frequency_for_states((victim, spectator))
        Omega = ion.get_rabi_rf(victim, spectator, 1e-6)
        sign = -1  # victim is the lower-energy state in this transition
        shift_lower += sign * ac_zeeman(Omega, w_transition, w_rf)

        spectator = ion.get_state_for_F(level, F=3, M_F=1)  # state 65
        w_transition = ion.get_transition_frequency_for_states((victim, spectator))
        Omega = ion.get_rabi_rf(victim, spectator, 1e-6)
        sign = -1  # victim is the lower-energy state in this transition
        shift_lower += sign * ac_zeeman(Omega, w_transition, w_rf)

        spectator = ion.get_state_for_F(level, F=5, M_F=1)  # state 79
        w_transition = ion.get_transition_frequency_for_states((victim, spectator))
        Omega = ion.get_rabi_rf(victim, spectator, 1e-6)
        sign = +1  # victim is the higher-energy state in this transition
        shift_lower += sign * ac_zeeman(Omega, w_transition, w_rf)

        spectator = ion.get_state_for_F(level, F=6, M_F=1)  # state 87
        w_transition = ion.get_transition_frequency_for_states((victim, spectator))
        Omega = ion.get_rabi_rf(victim, spectator, 1e-6)
        sign = +1  # victim is the higher-energy state in this transition
        shift_lower += sign * ac_zeeman(Omega, w_transition, w_rf)

        # shifts on upper state
        victim = upper
        shift_upper = 0.0

        spectator = ion.get_state_for_F(level, F=1, M_F=1)  # state 50
        w_transition = ion.get_transition_frequency_for_states((victim, spectator))
        Omega = ion.get_rabi_rf(victim, spectator, 1e-6)
        sign = -1  # victim is the lower-energy state in this transition
        shift_upper += sign * ac_zeeman(Omega, w_transition, w_rf)

        spectator = ion.get_state_for_F(level, F=2, M_F=1)  # state 57
        w_transition = ion.get_transition_frequency_for_states((victim, spectator))
        Omega = ion.get_rabi_rf(victim, spectator, 1e-6)
        sign = -1  # victim is the lower-energy state in this transition
        shift_upper += sign * ac_zeeman(Omega, w_transition, w_rf)

        spectator = ion.get_state_for_F(level, F=4, M_F=1)  # state 70
        w_transition = ion.get_transition_frequency_for_states((victim, spectator))
        Omega = ion.get_rabi_rf(victim, spectator, 1e-6)
        sign = +1  # victim is the higher-energy state in this transition
        shift_upper += sign * ac_zeeman(Omega, w_transition, w_rf)

        spectator = ion.get_state_for_F(level, F=5, M_F=1)  # state 79
        w_transition = ion.get_transition_frequency_for_states((victim, spectator))
        Omega = ion.get_rabi_rf(victim, spectator, 1e-6)
        sign = +1  # victim is the higher-energy state in this transition
        shift_upper += sign * ac_zeeman(Omega, w_transition, w_rf)

        spectator = ion.get_state_for_F(level, F=6, M_F=1)  # state 87
        w_transition = ion.get_transition_frequency_for_states((victim, spectator))
        Omega = ion.get_rabi_rf(victim, spectator, 1e-6)
        sign = +1  # victim is the higher-energy state in this transition
        shift_upper += sign * ac_zeeman(Omega, w_transition, w_rf)

        np.testing.assert_allclose(
            shift_upper - shift_lower,
            ac_zeeman_shift_for_transition(ion, (lower, upper), drive),
        )

        # calculate the shift on the same transition due to sigma+-polarized radiation
        drive = RFDrive(
            frequency=w_rf, amplitude=1e-6, polarization=SIGMA_PLUS_POLARIZATION
        )

        # shifts on lower-energy state
        victim = lower  # F=4, M_F=+1, state 70
        shift_lower = 0.0

        spectator = ion.get_state_for_F(level, F=2, M_F=2)  # state 51
        w_transition = ion.get_transition_frequency_for_states((victim, spectator))
        Omega = ion.get_rabi_rf(victim, spectator, 1e-6)
        sign = -1  # victim is the lower-energy state in this transition
        shift_lower += sign * ac_zeeman(Omega, w_transition, w_rf)

        spectator = ion.get_state_for_F(level, F=3, M_F=2)  # state 58
        w_transition = ion.get_transition_frequency_for_states((victim, spectator))
        Omega = ion.get_rabi_rf(victim, spectator, 1e-6)
        sign = -1  # victim is the lower-energy state in this transition
        shift_lower += sign * ac_zeeman(Omega, w_transition, w_rf)

        spectator = ion.get_state_for_F(level, F=4, M_F=2)  # state 67
        w_transition = ion.get_transition_frequency_for_states((victim, spectator))
        Omega = ion.get_rabi_rf(victim, spectator, 1e-6)
        sign = -1  # victim is the lower-energy state in this transition
        shift_lower += sign * ac_zeeman(Omega, w_transition, w_rf)

        spectator = ion.get_state_for_F(level, F=4, M_F=0)  # state 71
        w_transition = ion.get_transition_frequency_for_states((victim, spectator))
        Omega = ion.get_rabi_rf(victim, spectator, 1e-6)
        sign = +1  # victim is the higher-energy state in this transition
        shift_lower += sign * ac_zeeman(Omega, w_transition, w_rf)

        spectator = ion.get_state_for_F(level, F=5, M_F=0)  # state 81
        w_transition = ion.get_transition_frequency_for_states((victim, spectator))
        Omega = ion.get_rabi_rf(victim, spectator, 1e-6)
        sign = +1  # victim is the higher-energy state in this transition
        shift_lower += sign * ac_zeeman(Omega, w_transition, w_rf)

        spectator = ion.get_state_for_F(level, F=6, M_F=0)  # state 89
        w_transition = ion.get_transition_frequency_for_states((victim, spectator))
        Omega = ion.get_rabi_rf(victim, spectator, 1e-6)
        sign = +1  # victim is the higher-energy state in this transition
        shift_lower += sign * ac_zeeman(Omega, w_transition, w_rf)

        # shifts on upper state
        victim = upper  # F=3, M_F=+1, state 65
        shift_upper = 0.0

        spectator = ion.get_state_for_F(level, F=2, M_F=2)  # state 51
        w_transition = ion.get_transition_frequency_for_states((victim, spectator))
        Omega = ion.get_rabi_rf(victim, spectator, 1e-6)
        sign = -1  # victim is the lower-energy state in this transition
        shift_upper += sign * ac_zeeman(Omega, w_transition, w_rf)

        spectator = ion.get_state_for_F(level, F=3, M_F=2)  # state 58
        w_transition = ion.get_transition_frequency_for_states((victim, spectator))
        Omega = ion.get_rabi_rf(victim, spectator, 1e-6)
        sign = -1  # victim is the lower-energy state in this transition
        shift_upper += sign * ac_zeeman(Omega, w_transition, w_rf)

        spectator = ion.get_state_for_F(level, F=4, M_F=0)  # state 71
        w_transition = ion.get_transition_frequency_for_states((victim, spectator))
        Omega = ion.get_rabi_rf(victim, spectator, 1e-6)
        sign = +1  # victim is the higher-energy state in this transition
        shift_upper += sign * ac_zeeman(Omega, w_transition, w_rf)

        spectator = ion.get_state_for_F(level, F=5, M_F=0)  # state 81
        w_transition = ion.get_transition_frequency_for_states((victim, spectator))
        Omega = ion.get_rabi_rf(victim, spectator, 1e-6)
        sign = +1  # victim is the higher-energy state in this transition
        shift_upper += sign * ac_zeeman(Omega, w_transition, w_rf)

        spectator = ion.get_state_for_F(level, F=6, M_F=0)  # state 89
        w_transition = ion.get_transition_frequency_for_states((victim, spectator))
        Omega = ion.get_rabi_rf(victim, spectator, 1e-6)
        sign = +1  # victim is the higher-energy state in this transition
        shift_upper += sign * ac_zeeman(Omega, w_transition, w_rf)

        np.testing.assert_allclose(
            shift_upper - shift_lower,
            ac_zeeman_shift_for_transition(ion, (lower, upper), drive),
        )
