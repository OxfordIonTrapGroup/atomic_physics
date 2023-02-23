import numpy as np
from copy import deepcopy
import scipy.optimize as opt
import scipy.constants as consts
import atomic_physics as ap


_uB = consts.physical_constants["Bohr magneton"][0]
_uN = consts.physical_constants["nuclear magneton"][0]


def Lande_g(level: ap.Level):
    """Returns the Lande g factor for a level."""
    gL = 1
    gS = -consts.physical_constants["electron g factor"][0]

    S = level.S
    J = level.J
    L = level.L

    gJ = gL * (J * (J + 1) - S * (S + 1) + L * (L + 1)) / (2 * J * (J + 1)) + gS * (
        J * (J + 1) + S * (S + 1) - L * (L + 1)
    ) / (2 * J * (J + 1))
    return gJ


def df_dB(atom: ap.Atom, lower: int, upper: int, eps: float = 1e-6):
    """Returns the field-sensitivity of a transition between two
    states in the same level at a given magnetic field.

    :param atom: the atom
    :param lower: index of the lower energy state
    :param upper: index of the higher-energy state
    :param eps: field difference (T) to use when calculating derivatives
      numerically
    :return: the transition's field sensitivity (rad/s/T)

    To do: add a special case when we can use the BR formula
    """
    f = atom.delta(lower, upper)
    atom = deepcopy(atom)
    atom.setB(atom.B + eps)
    fpr = atom.delta(lower, upper)
    return (fpr - f) / eps


def d2f_dB2(atom: ap.Atom, lower: int, upper: int, eps: float = 1e-4):
    """Returns the second-order field-sensitivity of a transition
    between two states in thise same level at a given magnetic field.

    :param atom: the atom to work with
    :param lower: index of the lower energy state
    :param upper: index of the higher-energy state
    :param eps: field difference (T) to use when calculating derivatives
      numerically
    :return: the transition's second-order field sensitivity (Hz/s/T^2)

    To do: add a special case when we can use the BR formula
    """
    df = df_dB(atom, lower, upper)
    atom = deepcopy(atom)
    atom.setB(atom.B + eps)
    dfpr = df_dB(atom, lower, upper)

    return (dfpr - df) / eps


def field_insensitive_point(
    atom: ap.Atom, lower: int, upper: int, B_min: float = 1e-3, B_max: float = 1e-1
):
    """Returns the magnetic field at which the frequency of a transition
    between two states in the same level becomes first-order field independent.

    :param atom: the atom to work with
    :param lower: index of the lower energy state
    :param upper: index of the higher-energy state
    :param B_min: minimum magnetic field used in numerical minimization (T)
    :param B_max: maximum magnetic field used in numerical maximization (T)
    :return: the field-independent point (T) or None if none found

    To do: add special case where we can use the BR formula

    NB this does not change the atom's state (e.g. it does not set the B-field
    to the field insensitive point)
    """
    atom = deepcopy(atom)

    def fun(B: float):
        atom.setB(B)
        return df_dB(atom, lower, upper)

    res = opt.root(fun, x0=1e-8, options={"xtol": 1e-4, "eps": 1e-7})
    return res.x[0] if res.success else None


def ac_zeeman_shift(atom: ap.Atom, state: int, f_RF: float):
    """Returns the AC Zeeman shifts for a state normalized to a field of 1T.

    :param atom: the atom
    :param state: index of the state
    :param freq: frequency of the driving field (rad/s)
    :return: Array of AC Zeeman shifts (rad/s) caused by a field of 1T with
      sigma_minus, pi or sigma_plus polarisation [sigma_m, pi, sigma_p]
    """
    level = atom.level(state)
    states = atom.slice(level)
    state -= states.start

    E = atom.E[states]
    M = atom.M[states]
    R = atom.M1[states, states]
    rabi = R / consts.hbar

    acz = np.zeros(3)
    for q in [-1, 0, +1]:
        Mpr = M + q
        for _state in np.argwhere(M[state] == Mpr):
            freq = E[_state] - E[state]
            w = rabi[state, _state][0]
            acz[q + 1] += 0.5 * w ** 2 * (freq / (freq ** 2 - (f_RF) ** 2))
    return acz
