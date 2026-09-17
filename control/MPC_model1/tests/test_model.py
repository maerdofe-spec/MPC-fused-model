import unittest

import numpy as np

from MPC_model1.fossen_fixed_dl_model import (
    FixedLinearDampingRelativeModel,
)


class ModelTest(unittest.TestCase):
    def test_diagonal_closed_form_and_sign(self) -> None:
        mass = np.array([20.0, 25.0, 30.0])
        damping = np.array([8.0, 10.0, 12.0])
        dt = 0.1
        model = FixedLinearDampingRelativeModel(
            np.diag(mass), np.diag(damping), dt
        )
        expected_F = np.exp(-damping * dt / mass)
        expected_G = -(1.0 - expected_F) / damping
        np.testing.assert_allclose(np.diag(model.F), expected_F, rtol=1e-12)
        np.testing.assert_allclose(np.diag(model.G), expected_G, rtol=1e-12)

        _, velocity = model.predict_delta([1, 0, 0], [0, 0, 0], [2, 0, 0])
        self.assertLess(velocity[0], 0.0)


if __name__ == "__main__":
    unittest.main()
