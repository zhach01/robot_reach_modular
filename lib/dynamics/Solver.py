import sys, os

import numpy as np

sys.path.append(sys.path[0].replace(r"/lib/dynamics", r""))


def rungeKutta4(f: np.array, F: np.array, dt=0.001):
    """This function solves numerically a differential equation using Runge - Kutta (4th order)

    Args:
        f (np.array): differential equation (numerical)
        F (np.array): initial conditions (value already known by user)
        dt (float, optional): step size for calculation

    Returns:
        F (np.array): solution of differential equation
    """
    # Vectorized, allocation-free where possible.
    # NOTE:
    # - If `f` is a callable, we perform a true RK4 step with:
    #       k1 = f(F)
    #       k2 = f(F + dt/2 * k1)
    #       k3 = f(F + dt/2 * k2)
    #       k4 = f(F + dt   * k3)
    #       Fnext = F + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    # - If `f` is an array (as used elsewhere in this codebase: f = constant derivative at current state),
    #   a correct RK4 degenerates to a single Euler step: Fnext = F + dt * f.
    #   This branch keeps backward compatibility and fixes the previous per-element loop that produced
    #   incorrect weights.

    if callable(f):
        F = np.asarray(F, dtype=float)
        k1 = np.asarray(f(F), dtype=float)
        k2 = np.asarray(f(F + 0.5 * dt * k1), dtype=float)
        k3 = np.asarray(f(F + 0.5 * dt * k2), dtype=float)
        k4 = np.asarray(f(F + dt * k3), dtype=float)
        return F + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # f is an already-evaluated derivative array
    f_arr = np.asarray(f, dtype=float)
    F = np.asarray(F, dtype=float)
    return F + dt * f_arr


if __name__ == "__main__":

    """
    THIS SECTION IS FOR TESTING PURPOSES ONLY
    """

    print("Z")
