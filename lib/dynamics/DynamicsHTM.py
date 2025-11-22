# Access to parent folder to get its files
import sys, os

sys.path.append(sys.path[0].replace(r"/lib/dynamics", r""))

# Libraries
import numpy as np
from lib.kinematics.HTM import *
from lib.dynamics.Solver import *
from sympy import *
from lib.dynamics.fastsymp import fast_simplify, tidy, pinv_lr

# === Analytic Coriolis (Christoffel) helpers =================================
from sympy import Matrix, symbols, diff


# put this near the top (after imports)
def _ensure_symbolic_DH(robot):
    # Create the symbolic DH tables on demand
    if not hasattr(robot, "symbolicDHParameters"):
        robot.denavitHartenberg(symbolic=True)
    if not hasattr(robot, "symbolicDHParametersCOM"):
        robot.denavitHartenbergCOM(symbolic=True)


def _christoffel_C_from_D(Dsym: Matrix, qsym: Matrix, qdsym: Matrix):
    n = Dsym.shape[0]
    C = Matrix.zeros(n, n)
    for i in range(n):
        for j in range(n):
            cij = 0
            for k in range(n):
                cij += (
                    0.5
                    * (
                        diff(Dsym[i, j], qsym[k])
                        + diff(Dsym[i, k], qsym[j])
                        - diff(Dsym[j, k], qsym[i])
                    )
                    * qdsym[k]
                )
            C[i, j] = cij
    return tidy(C)


def inertiaMatrixCOM(robot: object, symbolic=False):
    """This function calculates the inertia matrix, with respect to each center of mass, given joints positions for dynamic model D(q) * q''(t) + C(q, q') * q'(t) + g(q) = τ

    Args:
        robot (object): serial robot (this won't work with other type of robots)
        symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
        D (NumPy Matrix): inertia matrix (numerical)
        D (SymPy Matrix): inertia matrix (symbolical)
    """

    # Calculate forward kinematics to each center of mass
    fkCOMHTM = forwardCOMHTM(robot, symbolic=symbolic)

    # Inertia matrix initialization
    D = (
        zeros(robot.jointsPositions.shape[0])
        if symbolic
        else np.zeros((robot.jointsPositions.shape[0], robot.jointsPositions.shape[0]))
    )

    # Iteration through each center of mass
    for j in range(len(robot.COMs)):

        # Velocity of each Center of Mass using Geometric Jacobian Matrix
        JgCOM = geometricJacobianCOM(robot, COM=j + 1, symbolic=symbolic)

        # Linear velocity mapping of current center of mass
        Jv = JgCOM[0:3, :]

        # Angular velocity mapping of current center of mass
        Jw = JgCOM[3:6, :]

        # Rotation matrix of the current Center of Mass (the sum is because of the way Python indexes)
        R = fkCOMHTM[j + 1][0:3, 0:3]

        # Inertia with respect to center of mass: Icom = R^T * I * R
        Icom = (
            R.T * robot.symbolicInertia[j] * R
            if symbolic
            else R.T.dot(robot.inertia[j]).dot(R)
        )

        # (m * Jv^T * JV) + (Jw^T * Icom * Jw)
        D += (
            (robot.symbolicMass[j] * (Jv.T * Jv)) + (Jw.T * Icom * Jw)
            if symbolic
            else (robot.mass[j] * (Jv.T.dot(Jv))) + (Jw.T.dot(Icom).dot(Jw))
        )

    return tidy(D) if symbolic else D


def inertiaMatrixCartesian(robot: object, symbolic=False):
    """This function calculates the inertia matrix, in cartesian space, given joints positions for dynamic model M(q) * x''(t) + N(q, q') * x'(t) + G(q) = τ

    Args:
        robot (object): serial robot (this won't work with other type of robots)
        symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
        M (NumPy Matrix): inertia matrix in cartesian space (numerical)
        M (SymPy Matrix): inertia matrix in cartesian space (symbolical)
    """

    # Inertia matrix calculation
    D = inertiaMatrixCOM(robot, symbolic)
    # Geometric Jacobian Matrix of the end - effector
    Jg = geometricJacobian(robot, symbolic)
    # Pseudo inverse of Geometric Jacobian Matrix
    Jinv = pinv_lr(Jg) if symbolic else np.linalg.pinv(Jg)
    # Calculation of Inertia Matrix in Cartesian Space
    M = (Jinv.T * D * Jinv) if symbolic else (Jinv.T).dot(D).dot(Jinv)
    return tidy(M) if symbolic else M


def kineticEnergyCOM(robot: object, symbolic=False):
    """This function calculates the total kinetic energy, with respect to each center of mass, given linear and angular velocities

    Args:
        robot (object): serial robot (this won't work with other type of robots)
        symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
        K (SymPy Matrix): kinetic matrix (symbolical)
    """

    # Kinetic Matrix calculation
    D = inertiaMatrixCOM(robot, symbolic)

    return (
        tidy(0.5 * (robot.qdSymbolic.T * D * robot.qdSymbolic))
        if symbolic
        else 0.5 * (robot.jointsVelocities.T.dot(D).dot(robot.jointsVelocities))
    )


def potentialEnergyCOM(
    robot: object, g=np.array([[0], [0], [-9.80665]]), symbolic=False
):
    """This function calculates the potential energy, with respect to each center of mass, given linear and angular velocities

    Args:
        robot (object): serial robot (this won't work with other type of robots)
        g (np.array, optional): gravity acceleration in Euclidian Space (R3)
        symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
        P (NumPy Array): potential energy (numerical)
        P (SymPy Matrix): potential energy (symbolical)
    """

    # Pose of each Center of Mass
    fkCOMHTM = forwardCOMHTM(robot, symbolic)

    # Potential energy initialization
    P = zeros(1) if symbolic else 0

    # Iteration through each center of mass
    for j in range(len(robot.COMs)):

        # Position of current Center of Mass (the sum is because of the way Python indexes)
        r = fkCOMHTM[j + 1][0:3, -1]

        # m * g^T * r
        P += (
            robot.symbolicMass[j] * ((g.T) * r)
            if symbolic
            else robot.mass[j] * ((g.T).dot(r))
        )

    return tidy(P) if symbolic else P


def gravitationalCOM(robot: object, g=np.array([[0], [0], [-9.80665]]), symbolic=False):
    """This function calculates the derivative of COMs' potential energy with respect to joints positions for dynamic model D(q) * q''(t) + C(q, q') * q'(t) + g(q) = τ

    Args:
        robot (object): serial robot (this won't work with other type of robots)
        g (np.array, optional): gravity acceleration in Euclidian Space (R3). Defaults to np.array([[0], [0], [-9.80665]]).
        symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
        G (NumPy Array): Gravitational effects' vector
        G (SymPy Matrix): Gravitational effects' vector
    """

    # Get number of joints
    n = robot.jointsPositions.shape[0]

    # Initialize differential term with zeros
    G = zeros(n, 1) if symbolic else np.zeros((n, 1))

    # Iteration through each center of mass
    for j in range(len(robot.COMs)):

        # Jacobian matrix to current center of mass
        JgCOM = geometricJacobianCOM(robot, COM=j + 1, symbolic=symbolic)

        # m * (JvCOM)^T * g
        G += (
            robot.symbolicMass[j] * JgCOM[0:3, :].T * g
            if symbolic
            else robot.mass[j] * (JgCOM[0:3, :].T).dot(g)
        )

    return tidy(G) if symbolic else G


def gravitationalCartesian(
    robot: object, g=np.array([[0], [0], [-9.80665]]), symbolic=False
):
    """This function calculates the gravitational effects that affect joints for the dynamic model M(q) * x''(t) + N(q, q') * x'(t) + G(q) = f

    Args:
        robot (object): serial robot (this won't work with other type of robots)
        g (np.array, optional): gravity acceleration in Euclidian Space (R3). Defaults to np.array([[0], [0], [-9.80665]]).
        symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
        G (NumPy Array): Gravitational effects' vector
        G (SymPy Matrix): Gravitational effects' vector
    """

    # Gravitational effects in joints space
    G = gravitationalCOM(robot, g, symbolic)

    # Geometric Jacobian Matrix of the end - effector
    Jg = geometricJacobian(robot, symbolic)

    # Use pseudo-inverse for robustness
    Jinv = fast_simplify(Jg.pinv()) if symbolic else np.linalg.pinv(Jg)
    G = fast_simplify(Jinv.T * G) if symbolic else Jinv.T.dot(G)

    return tidy(G) if symbolic else G


def centrifugalCoriolisCOMOLD(robot: object, dq=0.001, symbolic=False):
    """This function calculates the Centrifugal and Coriolis matrix for dynamic model

    Args:
        robot (object): serial robot (this won't work with other type of robots)
        dq (float, optional): step size for numeric derivative calculation. Defaults to 0.001
        symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
        C(q, q') (SymPy Matrix): Centrifugal and Coriolis matrix
    """

    # Inertia Matrix
    d = inertiaMatrixCOM(robot, symbolic)

    # Get number of joints (generalized coordinates)
    n = robot.jointsPositions.shape[0]

    # Initializes derivative matrix with zeros
    V = zeros(n) if symbolic else np.zeros((n, n))

    # Initialize Centrifugal and Coriolis matrix with zeros
    C = zeros(n) if symbolic else np.zeros((n, n))

    # Auxiliar variable to keep original joints positions
    q = robot.jointsPositions.copy()

    # Iterates through all colums of inertia matrix
    for j in range(n):

        # If symbolic calculation was requested
        if symbolic:

            # Differentiates current column with respect to joints positions
            V = d[:, j].jacobian(robot.qSymbolic)

        # Else, calculate derivative numerically
        else:

            # Iterates through all the generalized coordinates to calculate the derivative of current column
            for k in range(n):

                # Set increment to current generalized coordinate: z[j] = q[j] + dq
                robot.jointsPositions[k] += dq

                # Calculate inertia matrix with current step size
                D = inertiaMatrixCOM(robot)

                # Calculate derivative: [D[:, j](q + dq) - d[:, j](q)] / dq
                V[:, k] = (D[:, j] - d[:, j]) / dq

                # Eliminates step size by copying original values from auxiliar variable
                robot.jointsPositions[:, :] = q

        # Sum the previous derivative to get the "C" matrix and multiply it by qi'(t)
        C += (
            (V - (0.5 * V.T)) * robot.qdSymbolic[j]
            if symbolic
            else (V - (0.5 * V.T)) * robot.jointsVelocities[j]
        )

    return tidy(C) if symbolic else C


def centrifugalCoriolisCartesian(robot: object, dq=0.001, symbolic=False):
    """This function calculates the Centrifugal and Coriolis matrix in Cartesian Space

    Args:
        robot (object): serial robot (this won't work with other type of robots)
        dq (float, optional): step size for numeric derivative calculation. Defaults to 0.001
        symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
        N(q, q') (SymPy Matrix): Centrifugal and Coriolis Matrix in Cartesian Space
    """

    # Inertia Matrix in Cartesian Space
    M = inertiaMatrixCartesian(robot, symbolic)

    # Centrifugal and Coriolis Effects
    C = centrifugalCoriolisCOM(robot, dq, symbolic)

    # Geometric Jacobian Matrix of the end - effector
    Jg = geometricJacobian(robot, symbolic)

    # Time Derivative of Geometric Jacobian Matrix of the end - effector
    dJg = geometricJacobianDerivative(robot, symbolic)

    # Pseudo inverse of Geometric Jacobian Matrix
    Jinv = fast_simplify(Jg.pinv()) if symbolic else np.linalg.pinv(Jg)

    # Calculation of Coriolis Effects in Cartesian Space
    N = (
        (Jinv.T * C - M * dJg) * Jinv
        if symbolic
        else (Jinv.T.dot(C) - M.dot(dJg)).dot(Jinv)
    )

    return tidy(N) if symbolic else N


def centrifugalCoriolisCOM(
    robot: object, dq=0.001, symbolic=False, method: str = "finite_diff"
):
    """
    C(q, qdot): Coriolis/centrifugal matrix.
      method='analytic'   -> exact Christoffel from D(q)
      method='finite_diff' (default) -> your existing numeric derivative path
    """
    if method == "analytic":
        # Ensure robot has symbolic DH tables available
        _ensure_symbolic_DH(robot)

        # Build D(q) symbolically and then C(q,qdot)
        Dsym = inertiaMatrixCOM(robot, symbolic=True)
        qsym = robot.qSymbolic
        qdsym = robot.qdSymbolic
        Csym = _christoffel_C_from_D(Dsym, qsym, qdsym)

        if symbolic:
            return tidy(Csym)

        # Numeric substitution
        subs = {}
        n = robot.jointsPositions.shape[0]

        # joint positions & velocities
        for i in range(n):
            subs[qsym[i]] = float(robot.jointsPositions[i, 0])
            subs[qdsym[i]] = float(robot.jointsVelocities[i, 0])

        # link lengths and COMs (Serial defines these symbols)
        if hasattr(robot, "symbolicLinks"):
            for i in range(min(n, len(robot.symbolicLinks))):
                subs[robot.symbolicLinks[i]] = float(robot.linksLengths[i])
        if hasattr(robot, "symbolicCOMs"):
            for i in range(min(n, len(robot.symbolicCOMs))):
                subs[robot.symbolicCOMs[i]] = float(robot.COMs[i])

        # Optional: masses & inertias if your Robot exposes symbolic variants
        if hasattr(robot, "symbolicMass"):
            for i in range(len(robot.mass)):
                try:
                    subs[robot.symbolicMass[i]] = float(robot.mass[i])
                except:
                    pass
        if hasattr(robot, "symbolicInertia"):
            for j in range(len(robot.inertia)):
                try:
                    S = robot.symbolicInertia[j]
                    for r in range(3):
                        for c in range(3):
                            s = S[r, c]
                            if getattr(s, "free_symbols", None) or s != 0:
                                subs[s] = float(robot.inertia[j][r, c])
                except:
                    pass

        C_eval = Csym.subs(subs)
        return np.asarray(C_eval.evalf(), dtype=float)

    # === default: your existing finite-difference path ===
    d = inertiaMatrixCOM(robot, symbolic)
    n = robot.jointsPositions.shape[0]
    C = zeros(n, n) if symbolic else np.zeros((n, n))

    for j in range(n):
        if symbolic:
            V = d[:, j].jacobian(robot.qSymbolic)
        else:
            V = np.zeros((n, n))
            q_backup = robot.jointsPositions.copy()
            for k in range(n):
                robot.jointsPositions[k] += dq
                Dp = inertiaMatrixCOM(robot)
                V[:, k] = (Dp[:, j] - d[:, j]) / dq
                robot.jointsPositions[:, :] = q_backup
        C += (V - 0.5 * (V.T)) * (
            robot.qdSymbolic[j] if symbolic else robot.jointsVelocities[j]
        )

    return tidy(C) if symbolic else C


if __name__ == "__main__":

    """
    THIS SECTION IS FOR TESTING PURPOSES ONLY
    """

    print("Z")
