import unittest

import numpy as np

from atomic_physics.operators import (
    AngularMomentumLoweringOp,
    AngularMomentumProjectionOp,
    AngularMomentumRaisingOp,
    expectation_value,
)


class TestOperators(unittest.TestCase):
    def test_angular_momentum_ops(self):
        J = 5
        dim = 2 * J + 1
        focks = []
        for idx, M in enumerate(range(-5, 5 + 1)):
            focks.append(np.zeros(dim))
            focks[idx][idx] = 1

        for idx, M in enumerate(range(-5, 5 + 1)):
            np.testing.assert_allclose(
                AngularMomentumRaisingOp(j=J) @ focks[idx],
                np.sqrt((J - M) * (J + M + 1)) * focks[min(idx + 1, dim - 1)],
            )
            np.testing.assert_allclose(
                AngularMomentumLoweringOp(j=J) @ focks[idx],
                np.sqrt((J + M) * (J - M + 1)) * focks[max(idx - 1, 0)],
            )
            np.testing.assert_allclose(
                AngularMomentumProjectionOp(j=J) @ focks[idx], M * focks[idx]
            )

        np.testing.assert_allclose(
            AngularMomentumRaisingOp(J).T, AngularMomentumLoweringOp(J)
        )

    def test_expectation_value(self):
        J = 5
        dim = 2 * J + 1
        focks = np.diag(np.ones(dim))

        np.testing.assert_allclose(
            expectation_value(focks, AngularMomentumRaisingOp(j=J)),
            0.0,
        )
        np.testing.assert_allclose(
            expectation_value(focks, AngularMomentumLoweringOp(j=J)),
            0.0,
        )
        np.testing.assert_allclose(
            expectation_value(focks, AngularMomentumProjectionOp(j=J)),
            np.arange(-J, +J + 1),
        )
