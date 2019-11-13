import numpy as np
import scipy.optimize as opt
import scipy.constants as consts
import ion_phys.common as ip


def transition_freq(B, atom, level, lower, upper):
    """ Returns the frequency of a transition between two states in the same
    level at a given magnetic field.

    :param B: the static field (T)
    :param atom: the atom to work with
    :param level: the level the states are in
    :param lower: (F, MF) tuple for the lower-energy state
    :param upper: (F, MF) tuple for the higher-energy state
    :return: the transition frequency (Hz)
    """
    if isinstance(B, np.ndarray):
        B = B[0]

    ip.init(atom, B)
    E = level["E"]/consts.h
    M = level["M"]
    F = level["F"]
    return (E[np.logical_and(F == upper[0], M == upper[1])]
            - E[np.logical_and(F == lower[0], M == lower[1])])[0]


def df_dB(B, atom, level, lower, upper, eps=1e-4):
    """ Returns the field-sensitivity (Hz/T) of a transition between two states
    in the same level at a given magnetic field.

    :param B: the static field (T)
    :param atom: the atom to work with
    :param level: the level the states are in
    :param lower: (F, MF) tuple for the lower-energy state
    :param upper: (F, MF) tuple for the higher-energy state
    :param eps: field difference (T) to use when calculating derivatives
      numerically
    :return: the transition's field sensitivity (Hz/T)
    """
    return (transition_freq(B+eps, atom, level, lower, upper)
            - transition_freq(B, atom, level, lower, upper))/eps


def d2f_dB2(B, atom, level, lower, upper, eps=1e-4):
    """ Returns the second-order field-sensitivity (Hz/T^2) of a transition
    between two states in the same level at a given magnetic field.

    :param B: the static field (T)
    :param atom: the atom to work with
    :param level: the level the states are in
    :param lower: (F, MF) tuple for the lower-energy state
    :param upper: (F, MF) tuple for the higher-energy state
    :param eps: field difference (T) to use when calculating derivatives
      numerically
    :return: the transition's second-order field sensitivity (Hz/T^2)
    """
    return (df_dB(B+eps, atom, level, lower, upper)
            - df_dB(B, atom, level, lower, upper))/eps


def field_insensitive_point(atom, level, lower, upper, B_min=1e-3, B_max=1e-1):
    """ Returns the magnetic field at which the frequency of a transition
    between two states in the same level becomes first-order field independent.

    :param atom: the atom to work with
    :param level: the level the states are in
    :param lower: (F, MF) tuple for the lower-energy state
    :param upper: (F, MF) tuple for the higher-energy state
    :param B_min: minimum magnetic field used in numerical minimization (T)
    :param B_max: maximum magnetic field used in numerical maximization (T)
    :return: the field-independent point (T) or None if none found
    """
    res = opt.root(df_dB, x0=1e-8, args=(atom, level, lower, upper),
                   options={'xtol': 1e-4, 'eps': 1e-7})

    return res.x[0] if res.success else None


def ac_zeeman_shift(B, atom, level, state, freq):
    """ Returns the AC Zeeman shift of a particular state (Hz/T^2).
    :param B: the static field (T)
    :param atom: the atom to work with
    :param level: the level the states are in
    :param state: (F, MF) tuple of the state which the
        AC Zeeman shift is calculated
    :param freq: frequency of the driving field (Hz)
    :return: Array of AC Zeeman shift (Hz) caused by sigma_minus,
        pi and sigma_plus polarisations [sigma_m, pi, sigma_p]
    """
    ip.init(atom, B)
    ip.calc_m1(atom)

    M = level["M"]
    F = level["F"]
    R = level["R_m1"]
    rabi_freq = R/consts.hbar

    zm_shift = np.zeros(3)
    for f in np.unique(level["F"]):
        for m in np.arange(state[1] - 1, state[1] + 2):
            if abs(m) > f:
                continue
            else:
                q = m - state[1]
                trans_freq = 2*np.pi*transition_freq(B, atom, level, state, (f, m))
                w = rabi_freq[np.logical_and(F == state[0], M == state[1]),
                              np.logical_and(F == f, M == m)][0]
                zm_shift[q + 1] += 0.5*w**2*(trans_freq/(trans_freq**2 - (2*np.pi*freq)**2))
    return zm_shift/(2*np.pi)
