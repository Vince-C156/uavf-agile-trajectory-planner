import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now you can import functions from your_module.py
from integration_utils import runge_kutta_step
from dynamics import quad_dynamics
import numpy as np
import unittest
from scipy.integrate import solve_ivp

class TestDynamics(unittest.TestCase):
    def test_rk4_against_scipy(self):
        """
        Test the rk4 integrator against the scipy integrator.
        """
        inital_state = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1])
        dt = 1
        t = 0
        u = np.array([26, 26, 26, 26])
        quad = quad_dynamics()

        # Run the rk4 integrator
        rk4_state = runge_kutta_step(quad.calculate_dynamics, inital_state, t, dt, u)[1]

        # Run the scipy integrator
        scipy_sol = solve_ivp(quad.calculate_dynamics, [t, t+dt], inital_state, args=(u,), method='RK45')
        scipy_state = scipy_sol.y[:, -1]

        # Check that states are within 1e-6 of each other
        print(f'rk4_state: {rk4_state}')
        print(f'scipy_state: {scipy_state}')
        print(f'rk4_state - scipy_state: {np.linalg.norm(rk4_state - scipy_state)}')
        self.assertTrue(np.allclose(rk4_state, scipy_state, atol=1e-6))

if __name__ == '__main__':
    unittest.main()
