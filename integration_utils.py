import numpy as np
from typing import Tuple

def runge_kutta_step(f: callable, y0: float or np.ndarray, t0: float, dt: float, thrust: float or np.ndarray) -> Tuple[float, float or np.ndarray]:
    """
    Perform a single step of the fourth-order Runge-Kutta method.

    Parameters:
    - f: A function representing the derivative of the unknown function y(t).
    - y0: The initial value of y at t0.
    - t0: The current time.
    - h: The time step.
    - thrust: The instantaneous thrust value.

    Returns:
    - t1: The new time after the step.
    - y1: The new value of y after the step.
    """

    k1 = dt* f(t0, y0, thrust)
    k2 = dt* f(t0 + 0.5 * dt, y0 + 0.5 * k1, thrust)
    k3 = dt* f(t0 + 0.5 * dt, y0 + 0.5 * k2, thrust)
    k4 = dt* f(t0 + dt, y0 + k3, thrust)

    y1 = y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    t1 = t0 + dt

    return t1, y1