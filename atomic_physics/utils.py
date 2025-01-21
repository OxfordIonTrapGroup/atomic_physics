import numpy as np
import scipy.constants as consts
import scipy.optimize as opt

from atomic_physics.core import Atom, AtomFactory, Level, RFDrive, Transition
from atomic_physics.polarization import (
    cartesian_to_spherical,
    dM_for_transition,
)

_uB = consts.physical_constants["Bohr magneton"][0]
_uN = consts.physical_constants["nuclear magneton"][0]


def df_dB(
    atom_factory: AtomFactory,
    magnetic_field: float,
    states: tuple[int, int],
    eps: float = 1e-6,
) -> float:
    r"""Returns the field-sensitivity (:math:`\frac{\mathrm{d}f}{\mathrm{d}B}`)
    of a transition between two states in the same level at a given magnetic field.

    :param atom_factory: factory class for the atom of interest.
    :param magnetic_field: the magnetic to calculate the field sensitivity at.
    :param states: tuple of indices involved in this transition.
    :param eps: field difference (T) to use when calculating derivatives numerically.
    :return: the transition's field sensitivity (rad/s/T).
    """
    f_p = atom_factory(
        magnetic_field=magnetic_field + eps
    ).get_transition_frequency_for_states(states)
    f_m = atom_factory(
        magnetic_field=magnetic_field - eps
    ).get_transition_frequency_for_states(states)
    return (f_p - f_m) / (2 * eps)


def d2f_dB2(
    atom_factory: AtomFactory,
    magnetic_field: float,
    states: tuple[int, int],
    eps: float = 1e-6,
) -> float:
    r"""Returns the second-order field-sensitivity (:math:`\frac{\mathrm{d}^2f}{\mathrm{d}B^2}`)
    of a transition between two states in the same level at a given magnetic field.

    :param atom_factory: factory class for the atom of interest.
    :param magnetic_field: the magnetic to calculate the field sensitivity at.
    :param states: tuple of indices involved in this transition.
    :param eps: field difference (T) to use when calculating derivatives numerically.
    :return: the transition's field sensitivity (rad/s/T^2).
    """
    df_p = df_dB(atom_factory, magnetic_field + eps, states, eps)
    df_m = df_dB(atom_factory, magnetic_field - eps, states, eps)
    return (df_p - df_m) / (2 * eps)


def field_insensitive_point(
    atom_factory: AtomFactory,
    level_0: Level,
    F_0: float,
    M_F_0: float,
    level_1: Level,
    F_1: float,
    M_F_1: float,
    magnetic_field_guess: float = 1e-4,
    eps: float = 1e-4,
) -> float | None:
    """Returns the magnetic field at which the frequency of a transition
    between two states in the same level becomes first-order field independent.

    Since the energy ordering of states can change with magnetic field, we label states
    by ``F`` and ``M_F`` instead of using state indices.

    :param atom_factory: factory class for the atom of interest.
    :param level_0: level the first state involved in the transition lies in.
    :param F_0: value of ``F`` for the first state involved in the transition.
    :param M_F_0: value of ``M_F`` for the first state involved in the transition.
    :param level_1: level the second state involved in the transition lies in.
    :param F_1: value of ``F`` for the second state involved in the transition.
    :param M_F_1: value of ``M_F`` for the second state involved in the transition.
    :param magnetic_field_guess: Initial guess for the magnetic field insensitive point
        (T). This is used both as a seed for the root finding algorithm and as a scale
        factor to help numerical accuracy.
    :param eps: step size as a fraction of ``magnetic_field_guess`` to use when
        calculating numerical derivatives.
    :return: the field-independent point (T) or ``None`` if none found.
    """

    def opt_fun(x):
        magnetic_field = max(x, 10 * eps) * magnetic_field_guess
        atom = atom_factory(magnetic_field=magnetic_field)
        states = (
            atom.get_state_for_F(level=level_0, F=F_0, M_F=M_F_0),
            atom.get_state_for_F(level=level_1, F=F_1, M_F=M_F_1),
        )

        return df_dB(
            atom_factory,
            magnetic_field,
            states,
            eps=eps * magnetic_field_guess,
        )

    res = opt.root(
        opt_fun,
        x0=1,
        options={"xtol": 1e-4, "eps": eps},
    )

    return res.x[0] * magnetic_field_guess if res.success else None


def ac_zeeman_shift_for_state(atom: Atom, state: int, drive: RFDrive) -> float:
    r"""Returns the AC Zeeman shift on a given state resulting from an applied RF field.

    The calculated shift includes the counter-rotating term (Bloch-Siegert shift) and
    is calculated by summing
    :math:`\frac{1}{2} \Omega^2 \left(\omega_0 / (w_0^2 - w_{\mathrm{rf}}^2)\right)`
    over all magnetic dipole transitions which couple to the state. Where
    :math:`\Omega` is the Rabi frequency for the transition and :math:`w_0` is the
    transition frequency.

    Example:

    .. testcode::

        import numpy as np

        from atomic_physics.core import RFDrive
        from atomic_physics.ions.ca40 import Ca40, ground_level
        from atomic_physics.polarization import SIGMA_PLUS_POLARIZATION
        from atomic_physics.utils import ac_zeeman_shift_for_state

        ion = Ca40(magnetic_field=10e-4)
        w_transition = ion.get_transition_frequency_for_states(
            states=(
                ion.get_states_for_M(level=ground_level, M=+1 / 2),
                ion.get_states_for_M(level=ground_level, M=-1 / 2),
            )
        )
        w_rf = w_transition + 1e6 * 2 * np.pi  # RF is 1 MHz blue of the transition

        ac_zeeman_shift_for_state(
            atom=ion,
            state=ion.get_states_for_M(level=ground_level, M=+1 / 2),
            drive=RFDrive(frequency=w_rf, amplitude=1e-6, polarization=SIGMA_PLUS_POLARIZATION),
        )

    :param atom: the atom.
    :param state: index of the state to calculate the shift for.
    :param drive: the applied RF drive.
    :return: the AC Zeeman shift (rad/s).
    """
    level = atom.get_level_for_state(state)
    magnetic_dipoles = atom.get_magnetic_dipoles()
    Rnm = magnetic_dipoles / consts.hbar  # Rnm := (-1)**(q+1)<n|u_q|m>

    # Omega = B_{-q} * Rnm
    amplitude_for_pol = drive.amplitude * np.abs(
        cartesian_to_spherical(drive.polarization)[::-1]
    )

    acz = np.zeros(3)
    for delta_M in [-1, 0, +1]:
        spectators: list[int] = [
            spectator
            for spectator in atom.get_states_for_M(
                level=level, M=atom.M[state] + delta_M
            )
            if spectator != state
        ]
        for spectator in spectators:
            polarization = dM_for_transition(atom=atom, states=(state, spectator))
            w_transition = np.abs(
                atom.get_transition_frequency_for_states((state, spectator))
            )
            pol_ind = polarization + 1
            Omega = amplitude_for_pol[pol_ind] * Rnm[state, spectator]

            # A positive AC Zeeman shift means that the upper (higher energy) state
            # increases in energy, while the lower state decreases in energy
            sign = (
                +1
                if atom.state_energies[state] > atom.state_energies[spectator]
                else -1
            )

            acz[pol_ind] += sign * (
                0.5 * Omega**2 * (w_transition / (w_transition**2 - drive.frequency**2))
            )

    return sum(acz)


def ac_zeeman_shift_for_transition(
    atom: Atom, states: tuple[int, int], drive: RFDrive
) -> float:
    """Returns the AC Zeeman shift on a transition resulting from an applied RF field.

    See :func:`ac_zeeman_shift_for_state` for details.

    :param atom: the atom.
    :param states: tuple containing indices of the two states involved in the transition.
    :param drive: the applied RF drive.
    :return: the AC Zeeman shift (rad/s).
    """
    if len(states) != 2:
        raise ValueError(f"Expected 2 state indices, got {len(states)}.")

    upper = min(states)
    lower = max(states)

    return ac_zeeman_shift_for_state(
        atom=atom, state=upper, drive=drive
    ) - ac_zeeman_shift_for_state(atom=atom, state=lower, drive=drive)


def rayleigh_range(transition: Transition, waist_radius: float) -> float:
    """Returns the Rayleigh range for a given beam radius.

    :param transition: the transition the laser drives.
    :param waist_radius: Gaussian beam waist (:math:`1/e^2` intensity radius in m).
    :return: the Rayleigh range (m).
    """
    wavelength = consts.c / (transition.frequency / (2 * np.pi))
    return np.pi * waist_radius**2 / wavelength
