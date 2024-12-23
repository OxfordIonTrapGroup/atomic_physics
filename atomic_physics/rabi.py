from .common import Atom, State, Level
import numpy as np
from .beam import Beam
import scipy.constants as consts

__doc__ = """
    Calculate experimentally observable Rabi frequencies and light shifts (AC
    Stark shifts) in experimentally useful units. Essentially just does the relevant
    unit conversion from atom.ePole and takes into account the beam power and
    geometry.
    """


def scattering_amplitude_to_dipole_matrix_element_prefactor(wavelength: float):
    """
    Get the prefactor to convert a scattering amplitude (the scale that
    is used in atom.ePole) to a matrix element for a dipole transition.

    Taken from https://doi.org/10.1007/s003400050373

    :param wavelength: The wavelength of the transition.
    """
    return np.sqrt(
        3 * consts.epsilon_0 * consts.hbar * wavelength**3 / (8 * np.pi**2)
    )


def scattering_amplitude_to_quadrupole_matrix_element_prefactor(wavelength: float):
    """
    Get the prefactor to convert a scattering amplitude (the scale that
    is used in atom.ePole) to a matrix element for a quadrupole transition.

    Taken from https://doi.org/10.1007/s003400050373

    :param wavelength: The wavelength of the transition.
    """
    return np.sqrt(
        15 * consts.epsilon_0 * consts.hbar * wavelength**3 / (8 * np.pi**2)
    )


def laser_rabi_omega(state1: State, state2: State, atom: Atom, beam: Beam):
    """
    Return the (complex-valued) on-resonance Rabi frequency of the transition
    between state1 and state2 when driven by a laser beam in rad/s. The
    transition can be a dipole (E1) or quadrupole (E2) one.

    NOTE: The Rabi frequency returned has a phase arises from the Wigner3J
    symbols and the complex laser polarisation. This is useful to keep for
    more complex phase-dependent interactions such as light shifts and Raman
    transitions.

    TODO: The quadrupole Rabi frequency disagreed with the experimentally measured
    one in ABaQuS by a factor of ~2.3 in Dec 2024. Whether this is due to not fully
    known experimental params or errors in the maths is not clear. Check quadrupole
    Rabi frequency.

    :param state1: One of the atomic states in the transition.
    :param state2: The other atomic state in the transition.
    :param atom: The Atom object. ePole must have been pre-calculated.
    :param beam: Beam object containing the parameters of the driving
        beam.
    """

    assert (
        atom.ePole is not None
    ), "atom.ePole has not been calculated. Has the magnetic field been set?"
    idx1 = atom.index(state1.level, M=state1.M, F=state1.F)
    idx2 = atom.index(state2.level, M=state2.M, F=state2.F)
    if idx2 > idx1:
        idx_u = idx2
        idx_l = idx1
        state_u = state2
        state_l = state1
    else:
        idx_u = idx1
        idx_l = idx2
        state_u = state1
        state_l = state2

    q = state_u.M - state_l.M

    assert (state_l.level, state_u.level,) in atom.reverse_transitions.keys(), (
        "No dipole or quadrupole transition between"
        + f" {state_l.level} and {state_u.level} exists."
    )
    transition = atom.reverse_transitions[(state_l.level, state_u.level)]
    transition_freq = transition.freq + atom.delta(idx_l, idx_u)
    wavelength = 2 * np.pi * consts.c / transition_freq

    if abs(state_u.level.L - state_l.level.L) == 1:
        C = scattering_amplitude_to_dipole_matrix_element_prefactor(wavelength)
        if abs(q) > 1:
            return 0.0
        polarisation_coeff = beam.polarisation_rank1()[int(q + 1)]
    elif abs(state_u.level.L - state_l.level.L) == 2:
        C = scattering_amplitude_to_quadrupole_matrix_element_prefactor(wavelength)
        if abs(q) > 2:
            return 0.0
        polarisation_coeff = beam.polarisation_rank2()[int(q + 2)]
    else:
        raise ValueError(
            "Rabi frequencies for lasers can only be calculated for dipole or"
            + " quadrupole transitions. A transition with delta_L ="
            + f" {abs(state_u.level.L - state_l.level.L)} was provided."
        )

    prefactor = C * beam.E_field / consts.hbar * polarisation_coeff
    return prefactor * atom.ePole[idx_l, idx_u]


def raman_rabi_omega(
    state1: State,
    state2: State,
    atom: Atom,
    beam_red: Beam,
    beam_blue: Beam,
    virtual_levels: list[Level],
):
    """
    Return the (complex-valued) Rabi frequency  in rad/s of the transition between
    state1 and state2 when driven by a Raman transition. Assumes that these are
    within the same level.

    NOTE: The Rabi frequency returned has a phase arises from the Wigner3J
    symbols and the complex laser polarisation. This is useful to keep for
    more complex phase-dependent interactions such as light shifts.

    :param state1: One of the atomic states in the transition.
    :param state2: The other atomic state in the transition.
    :param atom: The Atom object. ePole must have been pre-calculated.
    :param beam_red: Beam object containing the parameters of the
        red-detuned driving beam.
    :param beam_red: Beam object containing the parameters of the
        blue-detuned driving beam.
    :param levels: List of all levels through which the virtual transition
        can occur (e.g. for transitions between two states in an S1/2 level,
        these would be the P1/2 and P3/2 levels). Only dipole transitions are
        considered for this.
    """

    assert (
        state1.level == state2.level
    ), "Raman transitions between states in different levels not supported"

    # It's important to determine which beam "drives" which virtual transition
    # out of state1 <-> virt. state and state2 <-> virt. state if the two beams
    # have different polarisations.
    # The red-detuned beam drives the transition from the higher of state1,2 and
    # the blue-detuned one drives the other one.
    idx_1 = atom.index(state1.level, M=state1.M, F=state1.F)
    idx_2 = atom.index(state2.level, M=state2.M, F=state2.F)
    if atom.E[idx_2] > atom.E[idx_1]:
        state_high = state2
        state_low = state1
    else:
        state_high = state1
        state_low = state2

    I = atom.I
    rabi_omega = 0

    for v_level in virtual_levels:
        # Check whether the virtual level is above or below the level that each
        # of the two states are in, to correctly determine the sign of delta_m
        # for each arm.
        if (state_high.level, v_level) in atom.reverse_transitions.keys():
            v_transition = atom.reverse_transitions[(state_high.level, v_level)]
        else:
            v_transition = atom.reverse_transitions[(v_level, state_high.level)]

        v_transition_delta = beam_red.omega - v_transition.freq

        v_J = v_level.J
        for F_int in np.arange(abs(I - v_J), I + v_J + 1):
            for m_int in np.arange(-F_int, F_int + 1):
                state_int = State(level=v_level, M=m_int, F=F_int)
                rabi_omega_red = laser_rabi_omega(state_high, state_int, atom, beam_red)
                rabi_omega_blue = laser_rabi_omega(
                    state_low, state_int, atom, beam_blue
                )
                rabi_omega += rabi_omega_red * rabi_omega_blue

        rabi_omega *= 1 / (4 * v_transition_delta)
    return rabi_omega


def far_detuned_light_shift(
    state: State,
    atom: Atom,
    beam: Beam,
    levels: list[Level],
):
    """
    Return the (complex) light shift frequency in rad/s on a state due to very
    far off-resonant transitions to a set of levels (which may occur when driving
    a Raman or quadrupole transition, for example).

    :param state: The State to calculate the light-shift for.
    :param beam: The Beam object that causes the light shift.
    :param atom: The Atom object. ePole must have been pre-calculated.
    :param levels: List of atomic levels that induce a light shift on ```state```.
    """
    state_level = state.level
    I = atom.I
    light_shift = 0
    # Since the transition is very off-resonant, say that all the transitions have the
    # same detuning
    laser_omega = beam.omega
    for aux_level in levels:
        transition = atom.reverse_transitions.get((state_level, aux_level), None)
        transition = atom.reverse_transitions.get((aux_level), transition)

        # Correct for sign if the state happens to be the top state
        if state_level == transition.lower:
            sign = 1
        else:
            sign = -1
        # Detuning has the sign such that a laser with lower frequency than the
        # transition pushes the state energy down, which occurs when detuning < 0
        detuning = laser_omega - transition.freq
        aux_J = aux_level.J
        for F_aux in np.arange(abs(I - aux_J), I + aux_J + 1):
            for m_aux in np.arange(-F_aux, F_aux + 1):
                delta_m = m_aux - state.M
                if abs(delta_m) <= 1:
                    state2 = State(level=aux_level, F=F_aux, M=m_aux)
                    rabi = laser_rabi_omega(state, state2, atom, beam)
                    light_shift += sign * abs(rabi) ** 2 / (4 * detuning)

    return light_shift
