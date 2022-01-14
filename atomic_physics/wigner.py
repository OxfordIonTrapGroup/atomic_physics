""" Wigner 3J symbols using the Racah formula. """
import numpy as np
from functools import lru_cache


_max_fact = 12  # 12 for int32, 20 for int64
_fact_store = np.ones(_max_fact + 1, dtype=np.int32)


for n in np.arange(_max_fact, dtype=np.int32):
    _fact_store[n + 1] = _fact_store[n] * np.int32(n + 1)


def _fact(n: int):
    """Returns n factorial."""
    assert n >= 0, str(n)
    return _fact_store[int(np.rint(n))]


def wigner3j(j1: float, j2: float, j3: float, m1: float, m2: float, m3: float):
    """Returns the Wigner 3J symbol."""

    # selection rules
    jT = np.int32(j1 + j2 + j3)
    if m1 + m2 + m3 != 0:
        return 0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0
    if j3 < np.abs(j1 - j2) or j3 > j1 + j2:
        return 0
    if jT != j1 + j2 + j3:
        return 0
    if not any([m1, m2]) and jT % 2 != 0:
        return 0

    # Use permutation relations to minimize required cache size:
    # - permute so that j1 <= j2 <= j3
    # - m1 >= 0
    sign = 0
    if j1 > j2:
        j1, j2 = j2, j1
        m1, m2 = m2, m1
        sign += 1
    if j2 > j3:
        j2, j3 = j3, j2
        m2, m3 = m3, m2
        sign += 1
    if m1 < 0:
        m1 = -m1
        m2 = -m2
        m3 = -m3
        sign += 1
    sign = (-1) ** (jT * sign)
    return sign * _wigner3j(j1, j2, j3, m1, m2, m3)


@lru_cache(maxsize=512)
def _wigner3j(j1: float, j2: float, j3: float, m1: float, m2: float, m3: float):
    sign = (-1) ** (np.abs(j1 - j2 - m3))
    tri = np.sqrt(
        (_fact(j1 + j2 - j3) * _fact(j1 + j3 - j2) * _fact(j2 + j3 - j1))
        / _fact(j1 + j2 + j3 + 1)
    )
    pre = np.sqrt(
        np.prod(
            [_fact(j + m) * _fact(j - m) for (j, m) in [(j1, m1), (j2, m2), (j3, m3)]]
        )
    )

    kmax = int(np.rint(min([j1 + j2 - j3, j1 - m1, j2 + m2])))
    kmin = int(np.rint(max([-(j3 - j2 + m1), -(j3 - j1 - m2), 0])))

    fact = 0
    for k in range(kmin, kmax + 1):
        fact += (-1) ** k / (
            _fact(k)
            * _fact(j1 + j2 - j3 - k)
            * _fact(j1 - m1 - k)
            * _fact(j2 + m2 - k)
            * _fact(j3 - j2 + m1 + k)
            * _fact(j3 - j1 - m2 + k)
        )
    return sign * tri * pre * fact
