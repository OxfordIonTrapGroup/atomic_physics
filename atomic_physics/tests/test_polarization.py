import unittest

import numpy as np

from atomic_physics.ions import ca43
from atomic_physics.polarization import (
    PI_POLARIZATION,
    SIGMA_MINUS_POLARIZATION,
    SIGMA_PLUS_POLARIZATION,
    X_POLARIZATION,
    Y_POLARIZATION,
    Z_POLARIZATION,
    cartesian_to_spherical,
    dM_for_transition,
    half_wave_plate,
    quarter_wave_plate,
    rotate_jones_matrix,
    spherical_dot,
    spherical_to_cartesian,
)


class TestPolarization(unittest.TestCase):
    def test_conversions(self):
        """Check conversions between Cartesian coordinates and the spherical basis."""
        for cartesian in (
            np.array((2.4, 0, 0)),
            np.array((0, 1.432, 0)),
            np.array((0, 0, 343)),
        ):
            np.testing.assert_allclose(
                cartesian, spherical_to_cartesian(cartesian_to_spherical(cartesian))
            )

    def test_spherical_dot(self):
        """Test that the spherical dot product gives the same result as the dot product
        in Cartesian coordinates.
        """
        vectors = (
            (np.array([1, 0, 0]), np.array([1, 0, 0])),
            (np.array([0, 1, 0]), np.array([0, 1, 0])),
            (np.array([0, 0, 1]), np.array([1, 0, 1])),
            (np.array([1, 0, 0]), np.array([0, 1, 0])),
            (np.array([1, 0, 1]), np.array([1, 0, 1])),
            (np.array([1, 1, 1]), np.array([1, 0, 1])),
        )
        for a, b in vectors:
            a_spher = cartesian_to_spherical(a)
            b_spher = cartesian_to_spherical(b)
            np.testing.assert_allclose(np.dot(a, b), spherical_dot(a_spher, b_spher))

    def test_dM_for_transition(self):
        """Check ``dM_for_transition``."""
        ion = ca43.Ca43(magnetic_field=146e-4)

        dM = dM_for_transition(
            atom=ion,
            states=(
                ion.get_state_for_F(level=ca43.D52, F=3, M_F=+1),
                ion.get_state_for_F(level=ca43.D52, F=4, M_F=+1),
            ),
        )
        assert np.isclose(dM, 0)

        dM = dM_for_transition(
            atom=ion,
            states=(
                ion.get_state_for_F(level=ca43.D52, F=3, M_F=0),  # state 67
                ion.get_state_for_F(level=ca43.D52, F=4, M_F=+1),  # state 74
            ),
        )
        assert np.isclose(dM, -1)

        dM = dM_for_transition(
            atom=ion,
            states=(
                ion.get_state_for_F(level=ca43.D52, F=4, M_F=+1),  # state 74
                ion.get_state_for_F(level=ca43.D52, F=3, M_F=0),  # state 67
            ),
        )
        assert np.isclose(dM, -1)

    def test_basis_vectors(self):
        """Check the pre-defined basis vectors are available and defined correctly."""
        np.testing.assert_allclose(X_POLARIZATION, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(Y_POLARIZATION, np.array([0, 1.0, 0.0]))
        np.testing.assert_allclose(Z_POLARIZATION, np.array([0, 0.0, 1.0]))

        np.testing.assert_allclose(
            SIGMA_MINUS_POLARIZATION,
            1 / np.sqrt(2) * (-X_POLARIZATION + 1j * Y_POLARIZATION),
        )
        np.testing.assert_allclose(
            SIGMA_PLUS_POLARIZATION,
            +1 / np.sqrt(2) * (X_POLARIZATION + 1j * Y_POLARIZATION),
        )
        np.testing.assert_allclose(Z_POLARIZATION, PI_POLARIZATION)

    def test_orthonormality(self):
        """Check the relationships between basis vectors in the spherical basis."""
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

    def test_jones_transformations(self):
        """Test functions which transform Jones matrices"""
        qwp = rotate_jones_matrix(quarter_wave_plate(), np.pi / 4)
        np.testing.assert_allclose(
            qwp @ X_POLARIZATION,
            SIGMA_PLUS_POLARIZATION,
        )
        qwp = rotate_jones_matrix(quarter_wave_plate(), np.pi + np.pi / 4)
        np.testing.assert_allclose(
            qwp @ X_POLARIZATION,
            SIGMA_PLUS_POLARIZATION,
        )
        qwp = rotate_jones_matrix(quarter_wave_plate(), -np.pi / 4)
        np.testing.assert_allclose(
            qwp @ Y_POLARIZATION,
            -1j * SIGMA_PLUS_POLARIZATION,
        )
        qwp = rotate_jones_matrix(quarter_wave_plate(), np.pi - np.pi / 4)
        np.testing.assert_allclose(
            qwp @ Y_POLARIZATION,
            -1j * SIGMA_PLUS_POLARIZATION,
            atol=1e-15,
        )

        qwp = rotate_jones_matrix(quarter_wave_plate(), -np.pi / 4)
        np.testing.assert_allclose(
            qwp @ X_POLARIZATION,
            -1 * SIGMA_MINUS_POLARIZATION,
        )
        qwp = rotate_jones_matrix(quarter_wave_plate(), np.pi - np.pi / 4)
        np.testing.assert_allclose(
            qwp @ X_POLARIZATION,
            -1 * SIGMA_MINUS_POLARIZATION,
            atol=1e-15,
        )
        qwp = rotate_jones_matrix(quarter_wave_plate(), +np.pi / 4)
        np.testing.assert_allclose(
            qwp @ Y_POLARIZATION,
            -1j * SIGMA_MINUS_POLARIZATION,
        )
        qwp = rotate_jones_matrix(quarter_wave_plate(), np.pi + np.pi / 4)
        np.testing.assert_allclose(
            qwp @ Y_POLARIZATION,
            -1j * SIGMA_MINUS_POLARIZATION,
            atol=1e-15,
        )

        hwp = half_wave_plate()
        np.testing.assert_allclose(hwp @ X_POLARIZATION, 1j * X_POLARIZATION)
        np.testing.assert_allclose(hwp @ Y_POLARIZATION, -1j * Y_POLARIZATION)

        hwp = rotate_jones_matrix(half_wave_plate(), np.pi)
        np.testing.assert_allclose(hwp, half_wave_plate(), atol=1e-15)

        hwp = rotate_jones_matrix(half_wave_plate(), np.pi / 2)
        np.testing.assert_allclose(
            hwp @ X_POLARIZATION, -1j * X_POLARIZATION, atol=1e-15
        )
        np.testing.assert_allclose(
            hwp @ Y_POLARIZATION, 1j * Y_POLARIZATION, atol=1e-15
        )

        hwp = rotate_jones_matrix(half_wave_plate(), np.pi / 4)
        np.testing.assert_allclose(
            hwp @ X_POLARIZATION, 1j * Y_POLARIZATION, atol=1e-15
        )
        np.testing.assert_allclose(
            hwp @ Y_POLARIZATION, 1j * X_POLARIZATION, atol=1e-15
        )
