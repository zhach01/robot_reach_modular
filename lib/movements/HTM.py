import sys, os
import numpy as np
from sympy import *

# keep parent access identical to original intent
sys.path.append(sys.path[0].replace(r"/lib/movements", r""))

# Small numeric identity cached to reduce allocations
_ID4 = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)


def tx(x=0, symbolic=False):
    """Translation on «x» axis

    Args:
      x (float or SymPy Symbol, optional): length of displacement in meters. Defaults to zero.
      symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
      H (np.array): Homogeneous Transformation Matrix (numerical)
      H (SymPy Matrix): Homogeneous Transformation Matrix (symbolical)
    """
    if symbolic:
        return Matrix([[1, 0, 0, x], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # fast path for zero
    if x == 0:
        return _ID4.copy()
    H = _ID4.copy()
    H[0, 3] = float(x)
    return H


def ty(y=0, symbolic=False):
    """Translation on «y» axis

    Args:
      y (float or SymPy Symbol, optional): length of displacement in meters. Defaults to zero.
      symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
      H (np.array): Homogeneous Transformation Matrix (numerical)
      H (SymPy Matrix): Homogeneous Transformation Matrix (symbolical)
    """
    if symbolic:
        return Matrix([[1, 0, 0, 0], [0, 1, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]])
    if y == 0:
        return _ID4.copy()
    H = _ID4.copy()
    H[1, 3] = float(y)
    return H


def tz(z=0, symbolic=False):
    """Translation on «z» axis

    Args:
      z (float or SymPy Symbol, optional): length of displacement in meters. Defaults to zero.
      symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
      H (np.array): Homogeneous Transformation Matrix (numerical)
      H (SymPy Matrix): Homogeneous Transformation Matrix (symbolical)
    """
    if symbolic:
        return Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, z], [0, 0, 0, 1]])
    if z == 0:
        return _ID4.copy()
    H = _ID4.copy()
    H[2, 3] = float(z)
    return H


def rx(x=0, symbolic=False):
    """Rotation on «x» axis

    Args:
      x (float or SymPy Symbol, optional): angle of rotation in radians. Defaults to zero.
      symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
      H (np.array): Homogeneous Transformation Matrix (numerical)
      H (SymPy Matrix): Homogeneous Transformation Matrix (symbolical)
    """
    if symbolic:
        return Matrix(
            [
                [1, 0, 0, 0],
                [0, cos(x), -sin(x), 0],
                [0, sin(x), cos(x), 0],
                [0, 0, 0, 1],
            ]
        )
    if x == 0:
        return _ID4.copy()
    c = float(np.cos(x))
    s = float(np.sin(x))
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def ry(y=0, symbolic=False):
    """Rotation on «y» axis

    Args:
      y (float or SymPy Symbol, optional): angle of rotation in radians. Defaults to zero.
      symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
      H (np.array): Homogeneous Transformation Matrix (numerical)
      H (SymPy Matrix): Homogeneous Transformation Matrix (symbolical)
    """
    if symbolic:
        return Matrix(
            [
                [cos(y), 0, sin(y), 0],
                [0, 1, 0, 0],
                [-sin(y), 0, cos(y), 0],
                [0, 0, 0, 1],
            ]
        )
    if y == 0:
        return _ID4.copy()
    c = float(np.cos(y))
    s = float(np.sin(y))
    return np.array(
        [
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def rz(z=0, symbolic=False):
    """Rotation on «z» axis

    Args:
      z (float or SymPy Symbol, optional): angle of rotation in radians. Defaults to zero.
      symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
      H (np.array): Homogeneous Transformation Matrix (numerical)
      H (SymPy Matrix): Homogeneous Transformation Matrix (symbolical)
    """
    if symbolic:
        return Matrix(
            [
                [cos(z), -sin(z), 0, 0],
                [sin(z), cos(z), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
    if z == 0:
        return _ID4.copy()
    c = float(np.cos(z))
    s = float(np.sin(z))
    return np.array(
        [
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def crossMatrix(r: np.array, symbolic=False):
    """Cross operator for three dimensional vectors

    Args:
        r (np.array  or SymPy Symbol): 3D vector
        symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
        c (np.array): Cross Operator (numeric)
        c (SymPy Matrix): Cross Operator Matrix (symbolic)
    """
    if symbolic:
        return Matrix([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
    r = np.asarray(r, dtype=float).reshape(-1)
    rx, ry_, rz_ = r[0], r[1], r[2]
    return np.array([[0.0, -rz_, ry_], [rz_, 0.0, -rx], [-ry_, rx, 0.0]], dtype=float)


if __name__ == "__main__":

    """
    THIS SECTION IS FOR TESTING PURPOSES ONLY
    """

    # Numerical representation of a translation
    H = tx(x=0.5)

    # Symbolical representation of a translation
    symbolicH = tx(x=1, symbolic=True)
    print("Z")
