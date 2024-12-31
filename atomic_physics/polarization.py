r"""Tools for working with polarization vectors.

See :ref:`definitions` for definitions and further discussion about representation of
polarization within ``atomic-physics``.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from atomic_physics.core import Atom


def cartesian_to_spherical(cartesian: np.ndarray) -> np.ndarray:
    """Converts a vector in Cartesian coordinates to the spherical basis.

    :param cartesian: input array in Cartesian coordinates.
    :return: the input array converted to the spherical basis.
    """
    if len(cartesian) != 3:
        raise ValueError(f"Expected a vector of length 3, got {cartesian.shape}")

    x = cartesian[0]
    y = cartesian[1]
    z = cartesian[2]

    A_p = -1 / np.sqrt(2) * (x + 1j * y)
    A_m = +1 / np.sqrt(2) * (x - 1j * y)
    A_0 = z

    return np.array((A_m, A_0, A_p))


def spherical_to_cartesian(spherical: np.ndarray) -> np.ndarray:
    """Converts a vector in the spherical basis to Cartesian coordinates.

    :param cartesian: input array in the spherical basis.
    :return: the input array converted to Cartesian coordinates.
    """
    if len(spherical) != 3:
        raise ValueError(f"Expected a vector of length 3, got {spherical.shape}")

    A_m = spherical[0]
    A_0 = spherical[1]
    A_p = spherical[2]

    x = 1 / np.sqrt(2) * (A_m - A_p)
    y = 1 / np.sqrt(2) * 1j * (A_m + A_p)
    z = A_0

    return np.array((x, y, z))


def spherical_dot(vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
    """Returns the dot product between two vectors in the spherical basis."""
    return (-vec_a[0] * vec_b[2]) + (vec_a[1] * vec_b[1]) + (-vec_a[2] * vec_b[0])


def retarder_jones(phi: float) -> np.ndarray:
    r"""Returns the Jones matrix for a field propagating along the z-axis passing
    through a retarder, whose fast axis is aligned along :math:`\hat{\mathbf{x}}`.

    :param phi: the phase shift added by the retarder.
    :returns: the Jones matrix.
    """
    return np.array([[np.exp(+0.5 * 1j * phi), 0], [0, np.exp(-0.5 * 1j * phi)]])


def rotate_jones_matrix(input_matrix: np.ndarray, theta: float) -> np.ndarray:
    """Rotates a Jones matrix about the z-axis (x-y plane rotation).

    the Jones matrix for a component rotated by an angle ``theta`` about the z-axis
    is given by ``R(theta) @ J @ R(-theta)``.

    :param input_array: Jones matrix to be rotated.
    :param theta: rotation angle (radians).
    :returns: the rotated Jones matrix.
    """
    R_theta = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    R_m_theta = np.array(
        [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    )
    return R_theta @ input_matrix @ R_m_theta


def half_wave_plate_jones() -> np.ndarray:
    r"""Returns the Jones matrix for a field propagating along the z-axis passing
    through a half-wave plate, whose fast axis is aligned along :math:`\hat{\mathbf{x}}`.
    """
    return retarder_jones(phi=np.pi)


def quarter_wave_plate_jones() -> np.ndarray:
    r"""Returns the Jones matrix for a field propagating along the z-axis passing
    through a quarter-wave plate, whose fast axis is aligned along
    :math:`\hat{\mathbf{x}}`.
    """
    return retarder_jones(phi=np.pi / 2)


def dM_for_transition(atom: "Atom", states: tuple[int, int]) -> int:
    r"""Returns the change in magnetic quantum number for a given transition.

    :math:`\delta M := M_{\mathrm{upper}} - M_{\mathrm{lower}}` where
    :math:`M_{\mathrm{upper}}` (:math:`M_{\mathrm{lower}}`) is
    the magnetic quantum number for the state with greater (lower) energy.

    :param atom: the atom.
    :param states: tuple containing indices of the two states involved in the transition.
    :return: the change in magnetic quantum number.
    """
    if len(states) != 2:
        raise ValueError(f"Expected 2 state indices, got {len(states)}.")

    # take advantage of the fact that states are ordered by energy
    upper = min(states)
    lower = max(states)
    return int(np.rint(atom.M[upper] - atom.M[lower]))


X_POLARIZATION: np.ndarray = np.array([1, 0, 0])
"""Jones vector for a field with linear polarization along the x-axis. """


Y_POLARIZATION: np.ndarray = np.array([0, 1, 0])
"""Jones vector for a field with linear polarization along the y-axis. """

Z_POLARIZATION: np.ndarray = np.array([0, 0, 1])
"""Jones vector for a field with linear polarization along the z-axis. """


PI_POLARIZATION = np.array([0, 0, 1])
r"""Jones vector for π polarization.

π-polarized radiation drives transitions where the magnetic quantum number is the same
in both states (:math:`M_{\mathrm{upper}} = M_{\mathrm{lower}}`).
"""


SIGMA_MINUS_POLARIZATION = spherical_to_cartesian(np.array([0, 0, 1]))
r"""Jones vector for σ- polarization.

σ- polarized radiation drives transitions where the magnetic quantum number in the
state with higher energy is 1 lower than in the state with lower energy (
:math:`M_{\mathrm{upper}} = M_{\mathrm{lower}} - 1`).
"""

SIGMA_PLUS_POLARIZATION = spherical_to_cartesian(np.array([1, 0, 0]))
r"""Jones vector for σ+ polarization.

σ+ polarized radiation drives transitions where the magnetic quantum number in the
state with higher energy is 1 greater than in the state with lower energy (
:math:`M_{\mathrm{upper}} = M_{\mathrm{lower}} + 1`).
"""
