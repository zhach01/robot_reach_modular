import numpy as np
from sympy import *
import sys, os

sys.path.append(sys.path[0].replace(r"/lib/movements", r""))

from lib.dynamics.fastsymp import tidy


def dqTx(x=0, symbolic=False):
    """Translation on «x» axis

    Args:
        x (float or SymPy Symbol, optional): length of displacement in meters. Defaults to zero.
        symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
        Q (np.array): Dual Quaternion (numeric)
        Q (SymPy Matrix): Dual Quaternion (symbolic)
    """
    if symbolic:
        return Matrix([[1], [0], [0], [0], [0], [0.5 * x], [0], [0]])
    return np.array(
        [[1.0], [0.0], [0.0], [0.0], [0.0], [0.5 * float(x)], [0.0], [0.0]], dtype=float
    )


def dqTy(y=0, symbolic=False):
    """Translation on «y» axis

    Args:
        y (float or SymPy Symbol, optional): length of displacement in meters. Defaults to zero.
        symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
        Q (np.array): Dual Quaternion (numeric)
        Q (SymPy Matrix): Dual Quaternion (symbolic)
    """
    if symbolic:
        return Matrix([[1], [0], [0], [0], [0], [0], [0.5 * y], [0]])
    return np.array(
        [[1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.5 * float(y)], [0.0]], dtype=float
    )


def dqTz(z=0, symbolic=False):
    """Translation on «z» axis

    Args:
        z (float or SymPy Symbol, optional): length of displacement in meters. Defaults to zero.
        symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
        Q (np.array): Dual Quaternion (numeric)
        Q (SymPy Matrix): Dual Quaternion (symbolic)
    """
    if symbolic:
        return Matrix([[1], [0], [0], [0], [0], [0], [0], [0.5 * z]])
    return np.array(
        [[1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.5 * float(z)]], dtype=float
    )


def dqRx(x=0, symbolic=False):
    """Rotation on «x» axis

    Args:
        x (float or SymPy Symbol, optional): angle of rotation in radians. Defaults to zero.
        symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
        Q (np.array): Dual Quaternion (numeric)
        Q (SymPy Matrix): Dual Quaternion (symbolic)
    """
    if symbolic:
        return Matrix([[cos(x / 2)], [sin(x / 2)], [0], [0], [0], [0], [0], [0]])
    c = np.cos(x * 0.5)
    s = np.sin(x * 0.5)
    return np.array([[c], [s], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype=float)


def dqRy(y=0, symbolic=False):
    """Rotation on «y» axis

    Args:
        y (float or SymPy Symbol, optional): angle of rotation in radians. Defaults to zero.
        symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
        Q (np.array): Dual Quaternion (numeric)
        Q (SymPy Matrix): Dual Quaternion (symbolic)
    """
    if symbolic:
        return Matrix([[cos(y / 2)], [0], [sin(y / 2)], [0], [0], [0], [0], [0]])
    c = np.cos(y * 0.5)
    s = np.sin(y * 0.5)
    return np.array([[c], [0.0], [s], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype=float)


def dqRz(z=0, symbolic=False):
    """Rotation on «z» axis

    Args:
        z (float or SymPy Symbol, optional): angle of rotation in radians. Defaults to zero.
        symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
        Q (np.array): Dual Quaternion (numeric)
        Q (SymPy Matrix): Dual Quaternion (symbolic)
    """
    if symbolic:
        return Matrix([[cos(z / 2)], [0], [0], [sin(z / 2)], [0], [0], [0], [0]])
    c = np.cos(z * 0.5)
    s = np.sin(z * 0.5)
    return np.array([[c], [0.0], [0.0], [s], [0.0], [0.0], [0.0], [0.0]], dtype=float)


def quaternionMultiplication(q1: np.array, q2: np.array, symbolic=False):
    """Quaternion multiplication: q = a * b

    Args:
        q1 (np.array) : quaternion
        q2 (np.array) : quaternion
        symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
        q (np.array): quaternion result (numeric)
        q (SymPy Matrix): quaternion result (symbolic)
    """
    if symbolic:
        # q assumed column Matrix 4x1
        r = (q1[0] * q2[0]) - (q1[1] * q2[1]) - (q1[2] * q2[2]) - (q1[3] * q2[3])
        i = (q1[0] * q2[1]) + (q1[1] * q2[0]) + (q1[2] * q2[3]) - (q1[3] * q2[2])
        j = (q1[0] * q2[2]) - (q1[1] * q2[3]) + (q1[2] * q2[0]) + (q1[3] * q2[1])
        k = (q1[0] * q2[3]) + (q1[1] * q2[2]) - (q1[2] * q2[1]) + (q1[3] * q2[0])
        return Matrix([r, i, j, k])
    a = np.asarray(q1, dtype=float).reshape(4)
    b = np.asarray(q2, dtype=float).reshape(4)
    r = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    i = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
    j = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
    k = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
    return np.array([[r], [i], [j], [k]], dtype=float)


def leftOperator(q: np.array, symbolic=False):
    """Left operator for Quaternions multiplication

    Args:
        q (np.array  or SymPy Symbol): Quaternion
        symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
        L (np.array): Left Operator (numeric)
        L (SymPy Matrix): Left Operator (symbolic)
    """
    if symbolic:
        a = Matrix([q[0], -q[1:, :]]).T
        b = q[1:, :].col_insert(1, (q[0] * eye(3)) + crossOperator(q, symbolic))
        return a.row_insert(1, b)
    q = np.asarray(q, dtype=float).reshape(4, 1)
    q0 = q[0, 0]
    v = q[1:].reshape(3, 1)
    L = np.empty((4, 4), dtype=float)
    L[0, 0] = q0
    L[0, 1:] = -v[:, 0]
    L[1:, 0] = v[:, 0]
    L[1:, 1:] = q0 * np.eye(3) + crossOperator(q, False)
    return L


def rightOperator(q: np.array, symbolic=False):
    """Right operator for Quaternions multiplication

    Args:
        q (np.array  or SymPy Symbol): Quaternion
        symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
        R (np.array): Right Operator (numeric)
        R (SymPy Matrix): Right Operator (symbolic)
    """
    if symbolic:
        a = Matrix([q[0], -q[1:, :]]).T
        b = q[1:, :].col_insert(1, (q[0] * eye(3)) - crossOperator(q, symbolic))
        return a.row_insert(1, b)
    q = np.asarray(q, dtype=float).reshape(4, 1)
    q0 = q[0, 0]
    v = q[1:].reshape(3, 1)
    R = np.empty((4, 4), dtype=float)
    R[0, 0] = q0
    R[0, 1:] = -v[:, 0]
    R[1:, 0] = v[:, 0]
    R[1:, 1:] = q0 * np.eye(3) - crossOperator(q, False)
    return R


def dqMultiplication(Qa: np.array, Qb: np.array, symbolic=False):
    """Dual Quaternion multiplication: Q = Qa * Qb

    Args:
        Qa (np.array) : Dual Quaternion
        Qb (np.array) : Dual Quaternion
        symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
        Q (np.array): Dual Quaternion result (numeric)
        Q (SymPy Matrix): Dual Quaternion result (symbolic)
    """
    if symbolic:
        r = quaternionMultiplication(Qa[0:4, :], Qb[0:4, :], True)
        x = quaternionMultiplication(Qa[0:4, :], Qb[4:8, :], True)
        y = quaternionMultiplication(Qa[4:8, :], Qb[0:4, :], True)
        return tidy(Matrix.vstack(r, x + y))
    r = quaternionMultiplication(Qa[0:4, :], Qb[0:4, :], False)
    x = quaternionMultiplication(Qa[0:4, :], Qb[4:8, :], False)
    y = quaternionMultiplication(Qa[4:8, :], Qb[0:4, :], False)
    return np.vstack((r, x + y))


def dualLeftOperator(Q: np.array, symbolic=False):
    """Left operator for Dual Quaternions multiplication

    Args:
        Q (np.array  or SymPy Symbol): Dual Quaternion
        symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
        L (np.array): Left Operator (numeric)
        L (SymPy Matrix): Left Operator (symbolic)
    """
    qr = Q[0:4, :]
    qd = Q[4:8, :]
    if symbolic:
        lr = leftOperator(qr, True)
        ld = leftOperator(qd, True)
        a = lr.col_insert(4, zeros(4))
        b = ld.col_insert(4, lr)
        return a.row_insert(4, b)
    lr = leftOperator(qr, False)
    ld = leftOperator(qd, False)
    top = np.append(lr, np.zeros((4, 4), dtype=float), axis=1)
    bot = np.append(ld, lr, axis=1)
    return np.append(top, bot, axis=0)


def dualRightOperator(Q: np.array, symbolic=False):
    """Right operator for Dual Quaternions multiplication

    Args:
        Q (np.array  or SymPy Symbol): Dual Quaternion
        symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
        R (np.array): Right Operator (numeric)
        R (SymPy Matrix): Right Operator (symbolic)
    """
    qr = Q[0:4, :]
    qd = Q[4:8, :]
    if symbolic:
        rr = rightOperator(qr, True)
        rd = rightOperator(qd, True)
        a = rr.col_insert(4, zeros(4))
        b = rd.col_insert(4, rr)
        return a.row_insert(4, b)
    rr = rightOperator(qr, False)
    rd = rightOperator(qd, False)
    top = np.append(rr, np.zeros((4, 4), dtype=float), axis=1)
    bot = np.append(rd, rr, axis=1)
    return np.append(top, bot, axis=0)


def crossOperator(q: np.array, symbolic=False):
    """Cross operator for quaternions' real part

    Args:
        q (np.array  or SymPy Symbol): quaternion
        symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
        c (np.array): Cross Operator (numeric)
        c (SymPy Matrix): Cross Operator Matrix (symbolic)
    """
    if symbolic:
        return Matrix(
            [[0.000, -q[3], +q[2]], [+q[3], 0.000, -q[1]], [-q[2], +q[1], 0.00]]
        )
    q = np.asarray(q, dtype=float).reshape(4)
    return np.array(
        [[0.0, -q[3], q[2]], [q[3], 0.0, -q[1]], [-q[2], q[1], 0.0]], dtype=float
    )


def dualCrossOperator(Q: np.array, symbolic=False):
    """Dual Cross operator for Dual Quaternions

    Args:
        Q (np.array  or SymPy Symbol): Dual Quaternion
        symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
        C (np.array): Cross Operator (numeric)
        C (SymPy Matrix): Cross Operator Matrix (symbolic)
    """
    Qr = crossOperatorExtension(Q[0:4, :], symbolic)
    Qd = crossOperatorExtension(Q[4:8, :], symbolic)
    if symbolic:
        a = Matrix([[Qr, zeros(4)]])
        b = Matrix([[Qd, Qr]])
        return Matrix([[a], [b]])
    top = np.append(Qr, np.zeros((4, 4), dtype=float), axis=1)
    bot = np.append(Qd, Qr, axis=1)
    return np.append(top, bot, axis=0)


def crossOperatorExtension(q: np.array, symbolic=False):
    """Cross operator extension for quaternions' multiplication

    Args:
        q (np.array  or SymPy Symbol): quaternion
        symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
        ce (np.array): Cross Operator Extension (numeric)
        ce (SymPy Matrix): Cross Operator Extension (symbolic)
    """
    if symbolic:
        return Matrix(
            [
                [0, 0.000, 0.000, 0.000],
                [0, 0.000, -q[3], +q[2]],
                [0, +q[3], 0.000, -q[1]],
                [0, -q[2], +q[1], 0.000],
            ]
        )
    q = np.asarray(q, dtype=float).reshape(4)
    return np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -q[3], q[2]],
            [0.0, q[3], 0.0, -q[1]],
            [0.0, -q[2], q[1], 0.0],
        ],
        dtype=float,
    )


def conjugateQ(Q: np.array, symbolic=False):
    """Conjugate operator for Quaternions

    Args:
        Q (np.array  or SymPy Symbol): Quaternion
        symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
        Q* (np.array): Conjugate Quaternion (numeric)
        Q* (SymPy Matrix): Conjugate Quaternion (symbolic)
    """
    if symbolic:
        return Matrix([[+Q[0, 0]], [-Q[1, 0]], [-Q[2, 0]], [-Q[3, 0]]])
    q = np.asarray(Q, dtype=float).reshape(4)
    return np.array([[q[0]], [-q[1]], [-q[2]], [-q[3]]], dtype=float)


def conjugateDQ(Q: np.array, symbolic=False):
    """Conjugate operator for Dual Quaternions

    Args:
        Q (np.array  or SymPy Symbol): Dual Quaternion
        symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
        Q* (np.array): Conjugate Dual Quaternion (numeric)
        Q* (SymPy Matrix): Conjugate Dual Quaternion (symbolic)
    """
    if symbolic:
        return Matrix(
            [
                [+Q[0, 0]],
                [-Q[1, 0]],
                [-Q[2, 0]],
                [-Q[3, 0]],
                [+Q[4, 0]],
                [-Q[5, 0]],
                [-Q[6, 0]],
                [-Q[7, 0]],
            ]
        )
    q = np.asarray(Q, dtype=float).reshape(8)
    return np.array(
        [[q[0]], [-q[1]], [-q[2]], [-q[3]], [q[4]], [-q[5]], [-q[6]], [-q[7]]],
        dtype=float,
    )


def dqToR3(Q: np.array, symbolic=False):
    """Transformation from Dual Quaternion to Euclidian Space Coordinates

    Args:
        Q (np.array  or SymPy Symbol): Dual Quaternion
        symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
        r (np.array): Position in R3 coordinates (numeric)
        r (SymPy Matrix): Position in R3 coordinates (symbolic)
    """
    if symbolic:
        qr = Matrix.vstack(Q[0:4, :], zeros(4, 1))
        r = 2 * dqMultiplication(Q, conjugateDQ(qr, True), True)
        return tidy(r[4:8, 0])
    qr = np.vstack(
        (
            np.asarray(Q[0:4, :], dtype=float).reshape(4, 1),
            np.zeros((4, 1), dtype=float),
        )
    )
    r = 2.0 * dqMultiplication(Q, conjugateDQ(qr, False), False)
    return r[4:8, 0].reshape(4, 1)


if __name__ == "__main__":

    """
    THIS SECTION IS FOR TESTING PURPOSES ONLY
    """

    # Numeric representation of a rotation
    Q = dualLeftOperator(dqRz(z=np.pi / 4)).dot(dqTx(x=0.5))

    # Fast dual quaternion multiplication
    Qmult = dqMultiplication(dqRz(z=np.pi / 4), dqTx(x=0.5))

    # Symbolic representation of a rotation
    symbolicQ = dqRz(z=Symbol("z"), symbolic=True)

    # Left Operator
    l = leftOperator(Q[0:4, :])

    # Right Operator
    r = rightOperator(symbolicQ[0:4, :], symbolic=True)

    # Dual Left Operator
    L = dualLeftOperator(Q)

    # Symbolic Dual Right Operator
    R = dualRightOperator(symbolicQ, symbolic=True)

    # Dual Cross Operator
    C = dualCrossOperator(Q)

    # From Dual Quaternion Space to Euclidian one
    r = dqToR3(Q)
    print(r)
    print("Z")
