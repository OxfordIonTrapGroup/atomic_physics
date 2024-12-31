import unittest

import numpy as np

from atomic_physics.polarization import (
    cartesian_to_spherical,
    spherical_dot,
    spherical_to_cartesian,
)


class TestPolarization(unittest.TestCase):
    def test_conversions(self):
        # Check the conversion functions self-invert correctly
        for cartesian in (
            np.array((2.4, 0, 0)),
            np.array((0, 1.432, 0)),
            np.array((0, 0, 343)),
        ):
            np.testing.assert_allclose(
                cartesian, spherical_to_cartesian(cartesian_to_spherical(cartesian))
            )

    def test_orthonormality(self):
        e_p = np.array([0, 0, 1])
        e_m = np.array([1, 0, 0])
        e_0 = np.array([0, 1, 0])

        np.testing.assert_allclose(spherical_dot(e_p, e_p), 0)
        np.testing.assert_allclose(spherical_dot(e_m, e_m), 0)
        np.testing.assert_allclose(spherical_dot(e_p, e_m), -1)
        np.testing.assert_allclose(spherical_dot(e_m, e_p), -1)
        np.testing.assert_allclose(spherical_dot(e_0, e_0), 1)
        np.testing.assert_allclose(spherical_dot(e_0, e_p), 0)
        np.testing.assert_allclose(spherical_dot(e_0, e_m), 0)
        np.testing.assert_allclose(
            spherical_to_cartesian(e_p), -spherical_to_cartesian(e_m).conj()
        )
        np.testing.assert_allclose(
            spherical_to_cartesian(e_m), -spherical_to_cartesian(e_p).conj()
        )

    # def test_jones(self):
    #     qwp = polarization.rotate_jones(polarization.quarter_wave_plate(), np.pi / 4)
    #     np.testing.assert_allclose(
    #         polarization.apply_jones_matrix(polarization.X_POLARIZATION, qwp),
    #         -1j * polarization.SIGMA_PLUS_POLARIZATION,
    #     )
    #     qwp = polarization.rotate_jones(
    #         polarization.quarter_wave_plate(), np.pi + np.pi / 4
    #     )
    #     np.testing.assert_allclose(
    #         polarization.apply_jones_matrix(polarization.X_POLARIZATION, qwp),
    #         -1j * polarization.SIGMA_PLUS_POLARIZATION,
    #         atol=1e-15,
    #     )
    #     qwp = polarization.rotate_jones(polarization.quarter_wave_plate(), -np.pi / 4)
    #     np.testing.assert_allclose(
    #         polarization.apply_jones_matrix(polarization.Y_POLARIZATION, qwp),
    #         -1 * polarization.SIGMA_PLUS_POLARIZATION,
    #     )
    #     qwp = polarization.rotate_jones(
    #         polarization.quarter_wave_plate(), np.pi - np.pi / 4
    #     )
    #     np.testing.assert_allclose(
    #         polarization.apply_jones_matrix(polarization.Y_POLARIZATION, qwp),
    #         -1 * polarization.SIGMA_PLUS_POLARIZATION,
    #         atol=1e-15,
    #     )

    #     qwp = polarization.rotate_jones(polarization.quarter_wave_plate(), -np.pi / 4)
    #     np.testing.assert_allclose(
    #         polarization.apply_jones_matrix(polarization.X_POLARIZATION, qwp),
    #         +1j * polarization.SIGMA_MINUS_POLARIZATION,
    #     )
    #     qwp = polarization.rotate_jones(
    #         polarization.quarter_wave_plate(), np.pi - np.pi / 4
    #     )
    #     np.testing.assert_allclose(
    #         polarization.apply_jones_matrix(polarization.X_POLARIZATION, qwp),
    #         +1j * polarization.SIGMA_MINUS_POLARIZATION,
    #         atol=1e-15,
    #     )
    #     qwp = polarization.rotate_jones(polarization.quarter_wave_plate(), +np.pi / 4)
    #     np.testing.assert_allclose(
    #         polarization.apply_jones_matrix(polarization.Y_POLARIZATION, qwp),
    #         -1 * polarization.SIGMA_MINUS_POLARIZATION,
    #     )
    #     qwp = polarization.rotate_jones(
    #         polarization.quarter_wave_plate(), np.pi + np.pi / 4
    #     )
    #     np.testing.assert_allclose(
    #         polarization.apply_jones_matrix(polarization.Y_POLARIZATION, qwp),
    #         -1 * polarization.SIGMA_MINUS_POLARIZATION,
    #         atol=1e-15,
    #     )
