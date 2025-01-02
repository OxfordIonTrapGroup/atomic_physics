import unittest

import numpy as np

from atomic_physics import polarization
from atomic_physics.core import RFDrive
from atomic_physics.ions import ba137, ca43
from atomic_physics.utils import ac_zeeman_shift_for_state


def ac_zeeman(Omega, w_transition, w_rf):
    return 0.5 * (Omega**2) * (w_transition / (w_transition**2 - w_rf**2))


class TestACZeeman(unittest.TestCase):
    def test_ac_zeeman_ground_level_ca43(self):
        """Check AC Zeeman shift calculations using the clock qubit in the 43Ca+ ground
        level.
        """
        ion = ca43.Ca43(magnetic_field=146.0942e-4)

        # clock transition
        upper_state_index = ion.get_state_for_F(level=ca43.S12, F=3, M_F=+1)
        lower_state_index = ion.get_state_for_F(level=ca43.S12, F=4, M_F=+0)
        w_transition = ion.get_transition_frequency_for_states(
            states=(upper_state_index, lower_state_index)
        )

        rf_amplitude = 1e-6
        w_rf = w_transition + 2 * np.pi * 1e6

        # Manually calculate the shifts for each polarization component
        #
        # Start by looking at the shift on the (3, +1) state
        victim_state_index = upper_state_index

        # pi shift
        spectator_state_index = ion.get_state_for_F(level=ca43.S12, F=4, M_F=+1)
        Omega = ion.get_rabi_m1(
            lower=spectator_state_index,
            upper=victim_state_index,
            amplitude=rf_amplitude,
        )
        w_spectator = ion.get_transition_frequency_for_states(
            states=(spectator_state_index, victim_state_index)
        )

        np.testing.assert_allclose(
            ac_zeeman(Omega, w_spectator, w_rf),
            ac_zeeman_shift_for_state(
                atom=ion,
                state=victim_state_index,
                drive=RFDrive(
                    frequency=w_rf,
                    amplitude=rf_amplitude,
                    polarization=polarization.PI_POLARIZATION,
                ),
            ),
        )

        # sigma- shift
        shift = 0.0
        for spectator_F, spectator_M in ((4, 2), (3, 0), (3, 2)):
            spectator_state_index = ion.get_state_for_F(
                level=ca43.S12, F=spectator_F, M_F=spectator_M
            )
            Omega = ion.get_rabi_m1(
                lower=spectator_state_index,
                upper=victim_state_index,
                amplitude=rf_amplitude,
            )
            w_spectator = ion.get_transition_frequency_for_states(
                states=(spectator_state_index, victim_state_index)
            )
            shift += ac_zeeman(Omega, w_spectator, w_rf)

        np.testing.assert_allclose(
            shift,
            ac_zeeman_shift_for_state(
                atom=ion,
                state=victim_state_index,
                drive=RFDrive(
                    frequency=w_rf,
                    amplitude=rf_amplitude,
                    polarization=polarization.SIGMA_MINUS_POLARIZATION,
                ),
            ),
        )

        # sigma+ shift
        spectator_state_index = ion.get_state_for_F(level=ca43.S12, F=4, M_F=0)
        Omega = ion.get_rabi_m1(
            lower=spectator_state_index,
            upper=victim_state_index,
            amplitude=rf_amplitude,
        )
        w_spectator = ion.get_transition_frequency_for_states(
            states=(spectator_state_index, victim_state_index)
        )
        shift = ac_zeeman(Omega, w_spectator, w_rf)

        np.testing.assert_allclose(
            shift,
            ac_zeeman_shift_for_state(
                atom=ion,
                state=victim_state_index,
                drive=RFDrive(
                    frequency=w_rf,
                    amplitude=rf_amplitude,
                    polarization=polarization.SIGMA_PLUS_POLARIZATION,
                ),
            ),
        )

        # Next look at the shift on the (4, 0) state
        victim_state_index = lower_state_index

        # pi shift
        spectator_state_index = ion.get_state_for_F(level=ca43.S12, F=3, M_F=0)
        Omega = ion.get_rabi_m1(
            lower=spectator_state_index,
            upper=victim_state_index,
            amplitude=rf_amplitude,
        )
        w_spectator = ion.get_transition_frequency_for_states(
            states=(spectator_state_index, victim_state_index)
        )

        np.testing.assert_allclose(
            ac_zeeman(Omega, w_spectator, w_rf),
            ac_zeeman_shift_for_state(
                atom=ion,
                state=victim_state_index,
                drive=RFDrive(
                    frequency=w_rf,
                    amplitude=rf_amplitude,
                    polarization=polarization.PI_POLARIZATION,
                ),
            ),
        )

        # sigma- shift
        spectator_state_index = ion.get_state_for_F(level=ca43.S12, F=3, M_F=-1)
        Omega = ion.get_rabi_m1(
            lower=spectator_state_index,
            upper=victim_state_index,
            amplitude=rf_amplitude,
        )
        w_spectator = ion.get_transition_frequency_for_states(
            states=(spectator_state_index, victim_state_index)
        )
        shift = ac_zeeman(Omega, w_spectator, w_rf)

        np.testing.assert_allclose(
            shift,
            ac_zeeman_shift_for_state(
                atom=ion,
                state=victim_state_index,
                drive=RFDrive(
                    frequency=w_rf,
                    amplitude=rf_amplitude,
                    polarization=polarization.SIGMA_MINUS_POLARIZATION,
                ),
            ),
        )

        # sigma+ shift
        shift = 0.0
        for spectator_F, spectator_M in ((3, +1), (4, -1), (4, 1)):
            spectator_state_index = ion.get_state_for_F(
                level=ca43.S12, F=spectator_F, M_F=spectator_M
            )
            Omega = ion.get_rabi_m1(
                lower=spectator_state_index,
                upper=victim_state_index,
                amplitude=rf_amplitude,
            )
            w_spectator = ion.get_transition_frequency_for_states(
                states=(spectator_state_index, victim_state_index)
            )
            shift += ac_zeeman(Omega, w_spectator, w_rf)

        np.testing.assert_allclose(
            shift,
            ac_zeeman_shift_for_state(
                atom=ion,
                state=victim_state_index,
                drive=RFDrive(
                    frequency=w_rf,
                    amplitude=rf_amplitude,
                    polarization=polarization.SIGMA_PLUS_POLARIZATION,
                ),
            ),
        )

    def test_ac_zeeman_D_level(self):
        """Check AC Zeeman shift calculations using a qubit in the metastable excited
        D5/2 level in 137Ba+.

        This test is a bit more interesting than the ground-level tests because there
        are more spectator transitions around
        """
        # Choose a low field so there is minimal state mixing and the standard selection
        # rules essentially apply, otherwise we'd have a lot more transitions to factor
        # in to our calculation!
        ion = ba137.Ba137(magnetic_field=0.1e-4)
        level = ba137.D52

        # Pick a pretty arbitrary choice of victim state, but one that has a number
        # of spectators around!
        victim_state_index = ion.get_state_for_F(level=level, F=3, M_F=+1)

        # Pick a sensible RF frequency by detuning 1MHz blue from a nearby transition
        ref = ion.get_state_for_F(level=level, F=2, M_F=1)
        w_rf = ion.get_transition_frequency_for_states((victim_state_index, ref)) * 1.1

        rf_amplitude = 1e-6

        # First look at pi polarized radiation
        shift = 0.0
        for spectator_F, spectator_M in ((4, +1), (2, +1)):
            spectator_state_index = ion.get_state_for_F(
                level=level, F=spectator_F, M_F=spectator_M
            )
            Omega = ion.get_rabi_m1(
                lower=spectator_state_index,
                upper=victim_state_index,
                amplitude=rf_amplitude,
            )
            w_spectator = ion.get_transition_frequency_for_states(
                states=(spectator_state_index, victim_state_index)
            )
            shift += ac_zeeman(Omega, w_spectator, w_rf)

        np.testing.assert_allclose(
            shift,
            ac_zeeman_shift_for_state(
                atom=ion,
                state=victim_state_index,
                drive=RFDrive(
                    frequency=w_rf,
                    amplitude=rf_amplitude,
                    polarization=polarization.PI_POLARIZATION,
                ),
            ),
        )

        # Now look at sigma+ transitions
        shift = 0.0
        for spectator_F, spectator_M in ((2, +2), (3, +2), (3, 0), (4, 0)):
            spectator_state_index = ion.get_state_for_F(
                level=level, F=spectator_F, M_F=spectator_M
            )
            Omega = ion.get_rabi_m1(
                lower=spectator_state_index,
                upper=victim_state_index,
                amplitude=rf_amplitude,
            )
            w_spectator = ion.get_transition_frequency_for_states(
                states=(spectator_state_index, victim_state_index)
            )
            shift += ac_zeeman(Omega, w_spectator, w_rf)

        np.testing.assert_allclose(
            shift,
            ac_zeeman_shift_for_state(
                atom=ion,
                state=victim_state_index,
                drive=RFDrive(
                    frequency=w_rf,
                    amplitude=rf_amplitude,
                    polarization=polarization.SIGMA_PLUS_POLARIZATION,
                ),
            ),
        )
