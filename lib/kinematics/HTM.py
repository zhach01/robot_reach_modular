# Access to parent folder to get its files
import sys, os

from pandas import array

sys.path.append(sys.path[0].replace(r"/lib/kinematics", r""))

# Libraries
import numpy as np
from lib.movements.HTM import *
from lib.dynamics.Solver import *
from sympy import *
from lib.dynamics.fastsymp import tidy


def forwardHTM(robot: object, symbolic=False):
    """Using Homogeneous Transformation Matrices, this function computes forward kinematics of a serial robot
    given joints positions in radians. Serial robot's kinematic parameters have to be set before using this function

    Args:
        robot (Serial): serial robot (this won't work with other type of robots)
        symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
        framesHTM (list): Homogeneous Transformation Matrices with frames' poses (numerical or symbolical)
    """

    # Initial conditions
    framesHTM = []

    if symbolic:
        # Get Denavit - Hartenberg Parameters Matrix
        DH = robot.symbolicDHParameters
    else:
        # Update Denavit - Hartenberg Parameters Matrix
        robot.denavitHartenberg()
        # Get Denavit - Hartenberg Matrix
        DH = robot.dhParameters

    # Create Homogeneous Transformation Matrix for Inertial Frame
    fkHTM = eye(4) if symbolic else np.identity(4)

    # Iteration through all the rows in Denavit - Hartenberg Matrix
    for frame in range(DH.rows) if symbolic else DH:
        # Operates matrices: Rz * Tz * Tx * Rx
        if symbolic:
            fkHTM = (
                fkHTM
                * rz(DH[frame, 0], True)
                * tz(DH[frame, 1], True)
                * tx(DH[frame, 2], True)
                * rx(DH[frame, 3], True)
            )
        else:
            fkHTM = fkHTM.dot(
                rz(frame[0]).dot(tz(frame[1])).dot(tx(frame[2])).dot(rx(frame[3]))
            )
        # Append each calculated Homogeneous Transformation Matrix
        framesHTM.append(fkHTM)

    return framesHTM


def forwardCOMHTM(robot: object, symbolic=False):
    """Using Homogeneous Transformation Matrices, this function computes forward kinematics of a serial robot's
    centers of mass given joints positions in radians. Serial robot's kinematic parameters have to be set before using this function

    Args:
      robot (Serial): serial robot (this won't work with other type of robots)
      symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
      framesHTM (list): Homogeneous Transformation Matrices with COMs' poses (numerical or symbolical)
    """

    # Calculate forward kinematics
    framesHTM = forwardHTM(robot, symbolic)

    # Initial conditions
    framesCOMHTM = [eye(4) if symbolic else np.identity(4)]

    if symbolic:
        # Get Denavit - Hartenberg Parameters Matrix
        comDH = robot.symbolicDHParametersCOM
    else:
        # Update Denavit - Hartenberg Parameters Matrix
        robot.denavitHartenbergCOM()
        # Get Denavit - Hartenberg Matrix
        comDH = robot.dhParametersCOM

    # Iteration through all the Centers of Mass
    for i in range(robot.symbolicCOMs.shape[0]):
        # Check where is the current Center of Mass
        rowCOM, column = robot.whereIsTheCOM(COM=i + 1)

        # Center of Mass Homogeneous Transformation Matrix
        if symbolic:
            COM = (
                rz(comDH[rowCOM, 0] if column >= 0 else 0, True)
                * tz(comDH[rowCOM, 1] if column >= 1 else 0, True)
                * tx(comDH[rowCOM, 2] if column >= 2 else 0, True)
                * rx(comDH[rowCOM, 3] if column >= 3 else 0, True)
            )
        else:
            COM = (
                rz(comDH[rowCOM, 0] if column >= 0 else 0)
                .dot(tz(comDH[rowCOM, 1] if column >= 1 else 0))
                .dot(tx(comDH[rowCOM, 2] if column >= 2 else 0))
                .dot(rx(comDH[rowCOM, 3] if column >= 3 else 0))
            )

        # Forward kinematics to Center of Mass
        fkCOMHTM = (
            (framesHTM[rowCOM - 1] * COM)
            if symbolic
            else framesHTM[rowCOM - 1].dot(COM)
        )

        # Append results
        framesCOMHTM.append(fkCOMHTM)

    return framesCOMHTM


def axisAngle(H: np.array, symbolic=False):
    """This function computes the axis - angle vector «X» using the Homogeneous Transformation Matrix of a reference frame

    Args:
      H (np.array): Homogeneous Transformation Matrix (numerical)
      symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
      X (NumPy Array): Axis - Angle vector (numerical)
      X (SymPy Matrix): Axis - Angle vector (symbolical)
    """

    if symbolic:
        # Calculate angle of rotation
        theta = acos((H[0:3, 0:3].trace() - 1) / 2)
        # Calculate axis of rotation
        n = (1 / (2 * sin(theta))) * Matrix(
            [[H[2, 1] - H[1, 2]], [H[0, 2] - H[2, 0]], [H[1, 0] - H[0, 1]]]
        )
        # Append position and orientation in one single vector
        X = H[0:3, 3].row_insert(3, theta * n)
        return tidy(X)
    else:
        # Calculate angle of rotation
        theta = np.arccos((np.trace(H[0:3, 0:3]) - 1) / 2)
        # Calculate axis of rotation
        n = (1 / (2 * np.sin(theta))) * np.array(
            [[H[2, 1] - H[1, 2]], [H[0, 2] - H[2, 0]], [H[1, 0] - H[0, 1]]]
        )
        # Append position and orientation in one single vector
        X = np.append(H[0:3, 3], theta * n)
        return X.reshape((6, 1))


def geometricJacobian(robot: object, symbolic=False):
    """Using Homogeneous Transformation Matrices, this function computes Geometric Jacobian Matrix of a serial robot
    given joints positions in radians. Serial robot's kinematic parameters have to be set before using this function

    Args:
      robot (Serial): serial robot (this won't work with other type of robots)
      symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
      J (np.array): Geometric Jacobian Matrix (numerical)
      J (SymPy Matrix): Geometric Jacobian Matrix (symbolical)
    """

    # Get number of joints (generalized coordinates)
    n = robot.jointsPositions.shape[0]

    # Calculate forward kinematics
    fkHTM = forwardHTM(robot, symbolic)

    # Initializes jacobian matrix with zeros
    J = zeros(6, n) if symbolic else np.zeros((6, n))

    # Iterates through all colums (generalized coordinates)
    for j in range(n):
        # Check in what row of Denavit Hartenberg Parameters Matrix is the current joint (the sum is because of the way Python indexes arrays)
        row, column = robot.whereIsTheJoint(j + 1)

        # Get row where joint is stored
        frame = (
            robot.symbolicDHParameters[4 * (row) : 4 * (row + 1)]
            if symbolic
            else robot.dhParameters[row, :]
        )

        # Get pose of the joint
        if symbolic:
            H = (
                fkHTM[row - 1]
                * rz(frame[0] if column >= 0 else 0, True)
                * tz(frame[1] if column >= 1 else 0, True)
                * tx(frame[2] if column >= 2 else 0, True)
                * rx(frame[3] if column >= 3 else 0, True)
            )
        else:
            H = (
                fkHTM[row - 1]
                .dot(rz(frame[0] if column >= 0 else 0))
                .dot(tz(frame[1] if column >= 1 else 0))
                .dot(tx(frame[2] if column >= 2 else 0))
                .dot(rx(frame[3] if column >= 3 else 0))
            )

        # Get axis of actuation of current joint
        z = H[0:3, 2]

        # Calculate distance between end - effector and current joint
        r = fkHTM[-1][0:3, 3] - H[0:3, 3]

        # Calculate axes of actuation of Center of Mass or End - Effector caused by current joint
        J[0:3, j] = z.cross(r) if symbolic else np.cross(z, r)

        # Set axis of actuation
        J[3:6, j] = z if symbolic else z

    return tidy(J) if symbolic else J


def geometricJacobianDerivative(robot: object, symbolic=False):
    """Using Homogeneous Transformation Matrices, this function computes the derivative of Geometric Jacobian Matrix of a
    serial robot given joints positions and velocities in radians. Serial robot's kinematic parameters have to be set before using this function

    Args:
      robot (Serial): serial robot (this won't work with other type of robots)
      symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
      dJ (np.array): Derivative of Geometric Jacobian Matrix (numerical)
      dJ (SymPy Matrix): Derivative of Geometric Jacobian Matrix (symbolical)
    """

    # Get number of joints (generalized coordinates)
    n = robot.jointsPositions.shape[0]

    # Calculate forward kinematics
    fkHTM = forwardHTM(robot, symbolic)

    # Define joints velocities
    qd = robot.qdSymbolic if symbolic else robot.jointsVelocities

    # Derivative of Jacobian Matrix
    dJ = zeros(6, n) if symbolic else np.zeros((6, n))

    # Iterates through all colums (generalized coordinates)
    for j in range(n):
        # Check in what row of Denavit Hartenberg Parameters Matrix is the current joint (the sum is because of the way Python indexes arrays)
        row, column = robot.whereIsTheJoint(j + 1)

        # Get row where joint is stored
        frame = (
            robot.symbolicDHParameters[4 * (row) : 4 * (row + 1)]
            if symbolic
            else robot.dhParameters[row, :]
        )

        # Get pose of the joint
        if symbolic:
            H = (
                fkHTM[row - 1]
                * rz(frame[0] if column >= 0 else 0, True)
                * tz(frame[1] if column >= 1 else 0, True)
                * tx(frame[2] if column >= 2 else 0, True)
                * rx(frame[3] if column >= 3 else 0, True)
            )
        else:
            H = (
                fkHTM[row - 1]
                .dot(rz(frame[0] if column >= 0 else 0))
                .dot(tz(frame[1] if column >= 1 else 0))
                .dot(tx(frame[2] if column >= 2 else 0))
                .dot(rx(frame[3] if column >= 3 else 0))
            )

        # Get the distance between the current joint and the end effector
        r = fkHTM[-1][0:3, -1] - H[0:3, -1]

        # Get axis of actuation of current joint
        z = H[0:3, 2]

        # Iterates through the remaining joints for the recursive sum for linear acceleration terms
        for i in range(j):
            # Check in what row of Denavit Hartenberg Parameters Matrix is the current joint (the sum is because of the way Python indexes arrays)
            rowi, columni = robot.whereIsTheJoint(i + 1)

            # Get row where joint is stored
            framei = (
                robot.symbolicDHParameters[4 * (rowi) : 4 * (rowi + 1)]
                if symbolic
                else robot.dhParameters[rowi, :]
            )

            # Get pose of the joint
            if symbolic:
                Hi = (
                    fkHTM[rowi - 1]
                    * rz(framei[0] if columni >= 0 else 0, True)
                    * tz(framei[1] if columni >= 1 else 0, True)
                    * tx(framei[2] if columni >= 2 else 0, True)
                    * rx(framei[3] if columni >= 3 else 0, True)
                )
            else:
                Hi = (
                    fkHTM[rowi - 1]
                    .dot(rz(framei[0] if columni >= 0 else 0))
                    .dot(tz(framei[1] if columni >= 1 else 0))
                    .dot(tx(framei[2] if columni >= 2 else 0))
                    .dot(rx(framei[3] if columni >= 3 else 0))
                )

            # Get axis of actuation of current joint
            zi = Hi[0:3, 2]

            # Set linear acceleration part
            if symbolic:
                dJ[0:3, j] += ((zi.cross(z)).cross(r) + zi.cross(z.cross(r))) * qd[i]
            else:
                dJ[0:3, j] += (
                    np.cross(np.cross(zi, z, axis=0), r, axis=0)
                    + np.cross(zi, np.cross(z, r, axis=0), axis=0)
                ) * qd[i]

        # Iterates through the remaining joints for the recursive sum for both linear and angular acceleration terms
        for i in range(j, n):
            # Check in what row of Denavit Hartenberg Parameters Matrix is the current joint (the sum is because of the way Python indexes arrays)
            rowi, columni = robot.whereIsTheJoint(i + 1)

            # Get row where joint is stored
            framei = (
                robot.symbolicDHParameters[4 * (rowi) : 4 * (rowi + 1)]
                if symbolic
                else robot.dhParameters[rowi, :]
            )

            # Get pose of the joint
            if symbolic:
                Hi = (
                    fkHTM[rowi - 1]
                    * rz(framei[0] if columni >= 0 else 0, True)
                    * tz(framei[1] if columni >= 1 else 0, True)
                    * tx(framei[2] if columni >= 2 else 0, True)
                    * rx(framei[3] if columni >= 3 else 0, True)
                )
            else:
                Hi = (
                    fkHTM[rowi - 1]
                    .dot(rz(framei[0] if columni >= 0 else 0))
                    .dot(tz(framei[1] if columni >= 1 else 0))
                    .dot(tx(framei[2] if columni >= 2 else 0))
                    .dot(rx(framei[3] if columni >= 3 else 0))
                )

            # Get axis of actuation of current joint
            zi = Hi[0:3, 2]

            # Relative position between i-th and i + 1 frames
            ri = fkHTM[-1][0:3, 3] - Hi[0:3, 3]

            # Set linear acceleration part
            if symbolic:
                dJ[0:3, j] += zi.cross(z.cross(ri)) * qd[i]
            else:
                dJ[0:3, j] += np.cross(zi, np.cross(z, ri, axis=0), axis=0) * qd[i]

            # Set angular acceleration part
            if symbolic:
                dJ[3:6, j] += z.cross(zi) * qd[i]
            else:
                dJ[3:6, j] += np.cross(z, zi, axis=0) * qd[i]

    return tidy(dJ) if symbolic else dJ


def geometricJacobianCOM(robot: object, COM: int, symbolic=False):
    """Using Homogeneous Transformation Matrices, this function computes Geometric Jacobian Matrix to a desired Center of Mass
    of a serial robot given joints positions in radians. Serial robot's kinematic parameters have to be set before using this function

    Args:
      robot (Serial): serial robot (this won't work with other type of robots)
      COM (int): center of mass that will be analyzed
      symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
      J (np.array): Geometric Jacobian Matrix (numerical)
      J (SymPy Matrix): Geometric Jacobian Matrix (symbolical)
    """

    # Get number of joints (generalized coordinates)
    n = robot.jointsPositions.shape[0]

    # Calculate forward kinematics
    fkHTM = forwardHTM(robot, symbolic)

    # Calculate forward kinematics to each Center of Mass
    fkCOMHTM = forwardCOMHTM(robot, symbolic)

    # Check in what row of Denavit Hartenberg Parameters Matrix is the Center of Mass (the sum is because of the way Python indexes arrays)
    rowCOM = robot.whereIsTheCOM(COM)[0]

    # Initializes jacobian matrix with zeros
    J = zeros(6, n) if symbolic else np.zeros((6, n))

    # Iterates through all colums (generalized coordinates)
    for i in range(n):
        # Check in what row of Denavit Hartenberg Parameters Matrix is the current joint (the sum is because of the way Python indexes arrays)
        row, column = robot.whereIsTheJoint(i + 1)

        # If the current joint is coupled after the desired center of mass, and any Center of Mass is analyzed
        if row > rowCOM:
            # Stop the algorithm, because current joint won't affect the desired center of mass
            break

        # Get row where joint is stored
        frame = (
            robot.symbolicDHParameters[4 * (row) : 4 * (row + 1)]
            if symbolic
            else robot.dhParameters[row, :]
        )

        # Get pose of joint
        if symbolic:
            H = (
                fkHTM[row - 1]
                * rz(frame[0] if column >= 0 else 0, True)
                * tz(frame[1] if column >= 1 else 0, True)
                * tx(frame[2] if column >= 2 else 0, True)
                * rx(frame[3] if column >= 3 else 0, True)
            )
        else:
            H = (
                fkHTM[row - 1]
                .dot(rz(frame[0] if column >= 0 else 0))
                .dot(tz(frame[1] if column >= 1 else 0))
                .dot(tx(frame[2] if column >= 2 else 0))
                .dot(rx(frame[3] if column >= 3 else 0))
            )

        # Get axis of actuation of current joint
        z = H[0:3, 2]

        # Calculate distance between Desired Center of Mass and current joint
        r = fkCOMHTM[COM][0:3, 3] - H[0:3, 3]

        # Calculate axes of actuation of Center of Mass or End - Effector caused by current joint
        J[0:3, i] = z.cross(r) if symbolic else np.cross(z, r)

        # Set axis of actuation
        J[3:6, i] = z if symbolic else z

    return tidy(J) if symbolic else J


def geometricJacobianDerivativeCOM(robot: object, COM: int, symbolic=False):
    """Using Homogeneous Transformation Matrices, this function computes the derivative of Geometric Jacobian Matrix of a serial robot
    given joints positions and velocities. Serial robot's kinematic parameters have to be set before using this function

    Args:
      robot (Serial): serial robot (this won't work with other type of robots)
      COM (int): center of mass that will be analyzed
      symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
      dJ (np.array): Derivative of Geometric Jacobian Matrix (numerical)
      dJ (SymPy Matrix): Derivative of Geometric Jacobian Matrix (symbolical)
    """

    # Get number of joints (generalized coordinates)
    n = robot.jointsPositions.shape[0]

    # Calculate forward kinematics
    fkHTM = forwardHTM(robot, symbolic)

    # Calculate forward kinematics to each Center of Mass
    fkCOMHTM = forwardCOMHTM(robot, symbolic)

    # Define joints velocities
    qd = robot.qdSymbolic if symbolic else robot.jointsVelocities

    # Check in what row of Denavit Hartenberg Parameters Matrix is the Center of Mass
    rowCOM = robot.whereIsTheCOM(COM)[0]

    # Derivative of Jacobian Matrix
    dJ = zeros(6, n) if symbolic else np.zeros((6, n))

    # Iterates through all colums (generalized coordinates)
    for j in range(n):
        # Check in what row of Denavit Hartenberg Parameters Matrix is the current joint (the sum is because of the way Python indexes arrays)
        row, column = robot.whereIsTheJoint(j + 1)

        # If the current joint is coupled after the desired center of mass, and any Center of Mass is analyzed
        if row > rowCOM:
            # Stop the algorithm, because current joint won't affect the desired center of mass
            break

        # Get row where joint is stored
        frame = (
            robot.symbolicDHParameters[4 * (row) : 4 * (row + 1)]
            if symbolic
            else robot.dhParameters[row, :]
        )

        # Get pose of the joint
        if symbolic:
            H = (
                fkHTM[row - 1]
                * rz(frame[0] if column >= 0 else 0, True)
                * tz(frame[1] if column >= 1 else 0, True)
                * tx(frame[2] if column >= 2 else 0, True)
                * rx(frame[3] if column >= 3 else 0, True)
            )
        else:
            H = (
                fkHTM[row - 1]
                .dot(rz(frame[0] if column >= 0 else 0))
                .dot(tz(frame[1] if column >= 1 else 0))
                .dot(tx(frame[2] if column >= 2 else 0))
                .dot(rx(frame[3] if column >= 3 else 0))
            )

        # Get the distance between the current joint and the end effector
        r = fkCOMHTM[COM][0:3, -1] - H[0:3, -1]

        # Get axis of actuation of current joint
        z = H[0:3, 2]

        # Iterates through the remaining joints for the recursive sum for linear acceleration terms
        for i in range(j):
            # Check in what row of Denavit Hartenberg Parameters Matrix is the current joint (the sum is because of the way Python indexes arrays)
            rowi, columni = robot.whereIsTheJoint(i + 1)

            # If the current frame is coupled after the desired center of mass, and any Center of Mass is analyzed
            if rowi > rowCOM:
                # Stop the algorithm, because current joint won't affect the desired center of mass
                break

            # Get row where joint is stored
            framei = (
                robot.symbolicDHParameters[4 * (rowi) : 4 * (rowi + 1)]
                if symbolic
                else robot.dhParameters[rowi, :]
            )

            # Get pose of the joint
            if symbolic:
                Hi = (
                    fkHTM[rowi - 1]
                    * rz(framei[0] if columni >= 0 else 0, True)
                    * tz(framei[1] if columni >= 1 else 0, True)
                    * tx(framei[2] if columni >= 2 else 0, True)
                    * rx(framei[3] if columni >= 3 else 0, True)
                )
            else:
                Hi = (
                    fkHTM[rowi - 1]
                    .dot(rz(framei[0] if columni >= 0 else 0))
                    .dot(tz(framei[1] if columni >= 1 else 0))
                    .dot(tx(framei[2] if columni >= 2 else 0))
                    .dot(rx(framei[3] if columni >= 3 else 0))
                )

            # Get axis of actuation of current joint
            zi = Hi[0:3, 2]

            # Set linear acceleration part
            if symbolic:
                dJ[0:3, j] += ((zi.cross(z)).cross(r) + zi.cross(z.cross(r))) * qd[i]
            else:
                dJ[0:3, j] += (
                    np.cross(np.cross(zi, z, axis=0), r, axis=0)
                    + np.cross(zi, np.cross(z, r, axis=0), axis=0)
                ) * qd[i]

        # Iterates through the remaining joints for the recursive sum for both linear and angular acceleration terms
        for i in range(j, n):
            # Check in what row of Denavit Hartenberg Parameters Matrix is the current joint (the sum is because of the way Python indexes arrays)
            rowi, columni = robot.whereIsTheJoint(i + 1)

            # If the current frame is coupled after the desired center of mass, and any Center of Mass is analyzed
            if rowi > rowCOM:
                # Stop the algorithm, because current joint won't affect the desired center of mass
                break

            # Get row where joint is stored
            framei = (
                robot.symbolicDHParameters[4 * (rowi) : 4 * (rowi + 1)]
                if symbolic
                else robot.dhParameters[rowi, :]
            )

            # Get pose of the joint
            if symbolic:
                Hi = (
                    fkHTM[rowi - 1]
                    * rz(framei[0] if columni >= 0 else 0, True)
                    * tz(framei[1] if columni >= 1 else 0, True)
                    * tx(framei[2] if columni >= 2 else 0, True)
                    * rx(framei[3] if columni >= 3 else 0, True)
                )
            else:
                Hi = (
                    fkHTM[rowi - 1]
                    .dot(rz(framei[0] if columni >= 0 else 0))
                    .dot(tz(framei[1] if columni >= 1 else 0))
                    .dot(tx(framei[2] if columni >= 2 else 0))
                    .dot(rx(framei[3] if columni >= 3 else 0))
                )

            # Get axis of actuation of current joint
            zi = Hi[0:3, 2]

            # Relative position between i-th and i + 1 frames
            ri = fkCOMHTM[COM][0:3, 3] - Hi[0:3, 3]

            # Set linear acceleration part
            if symbolic:
                dJ[0:3, j] += zi.cross(z.cross(ri)) * qd[i]
            else:
                dJ[0:3, j] += np.cross(zi, np.cross(z, ri, axis=0), axis=0) * qd[i]

            # Set angular acceleration part
            if symbolic:
                dJ[3:6, j] += z.cross(zi) * qd[i]
            else:
                dJ[3:6, j] += np.cross(z, zi, axis=0) * qd[i]

    return tidy(dJ) if symbolic else dJ


def analyticJacobian(robot: object, dq=0.001, symbolic=False):
    """Using Homogeneous Transformation Matrices, this function computes Analytic Jacobian Matrix of a serial robot
    given joints positions in radians. Serial robot's kinematic parameters have to be set before using this function

    Args:
      robot (Serial): serial robot (this won't work with other type of robots)
      dq (float, optional): step size for numerical derivative. Defaults to 0.001.
      symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
      J (np.array): Inertial Analytic Jacobian Matrix (numerical)
      J (SymPy Matrix): Inertial Analytic Jacobian Matrix (symbolical)
    """

    # Calculate forward kinematics: f(q)
    fkHTM = forwardHTM(robot, symbolic)

    # Convert result into an Axis - Angle vector: x(q)
    x = axisAngle(fkHTM[-1], symbolic)

    if symbolic:
        # Calculate Analytic Jacobian Matrix by differentiating Axis - Angle vector with SymPy functions
        return tidy(x.jacobian(robot.qSymbolic))
    else:
        # Get number of joints (generalized coordinates)
        n = robot.jointsPositions.shape[0]

        # Initializes jacobian matrix with zeros
        J = np.zeros((6, n))

        # Auxiliar variable to keep original joints positions
        q = robot.jointsPositions.copy()

        # Iterates through all colums (generalized coordinates)
        for j in range(n):
            # Set increment to current generalized coordinate: z[j] = q[j] + dq
            robot.jointsPositions[j] += dq

            # Calculate forward kinematics with step size: f(z) = f(q + dq)
            f = forwardHTM(robot)

            # Convert result into an Axis - Angle vector: X(q + dq)
            X = axisAngle(f[-1])

            # Calculate analytic jacobian matrix: [X(q + dq) - x(q)] / dq
            J[:, j] = ((X - x) / dq).flatten()

            # Eliminates step size by copying original values from auxiliar variable
            robot.jointsPositions[:, :] = q

        return J


def analyticJacobianCOM(robot: object, COM: int, dq=0.001, symbolic=False):
    """Using Homogeneous Transformation Matrices, this function computes Analytic Jacobian Matrix to a Center of Mass
    of a serial robot given joints positions in radians. Serial robot's kinematic parameters have to be set before using this function

    Args:
      robot (Serial): serial robot (this won't work with other type of robots)
      COM (int): center of mass that will be analyzed
      dq (float, optional): step size for numerical derivative. Defaults to 0.001.
      symbolic (bool, optional): used to calculate symbolic equations. Defaults to False.

    Returns:
      J (np.array): Inertial Analytic Jacobian Matrix to Center of Mass (numerical)
      J (SymPy Matrix): Inertial Analytic Jacobian Matrix to Center of Masas (symbolical)
    """

    # Calculate forward kinematics: f(q)
    fkCOMHTM = forwardCOMHTM(robot, symbolic)

    # Convert result into an Axis - Angle vector: x(q)
    x = axisAngle(fkCOMHTM[COM], symbolic)

    if symbolic:
        # Calculate Analytic Jacobian Matrix by differentiating Axis - Angle vector with SymPy functions
        return tidy(x.jacobian(robot.qSymbolic))
    else:
        # Get number of joints (generalized coordinates)
        n = robot.jointsPositions.shape[0]

        # Initializes jacobian matrix with zeros
        J = np.zeros((6, n))

        # Auxiliar variable to keep original joints positions
        q = robot.jointsPositions.copy()

        # Iterates through all colums (generalized coordinates)
        for j in range(n):
            # Set increment to current generalized coordinate: z[j] = q[j] + dq
            robot.jointsPositions[j] += dq

            # Calculate forward kinematics with step size: f(z) = f(q + dq)
            f = forwardCOMHTM(robot)

            # Convert result into an Axis - Angle vector: X(q + dq)
            X = axisAngle(f[COM])

            # Calculate analytic jacobian matrix: [X(q + dq) - x(q)] / dq
            J[:, j] = ((X - x) / dq).flatten()

            # Eliminates step size by copying original values from auxiliar variable
            robot.jointsPositions[:, :] = q

        return J


def inverseHTM(
    robot: object, q0: np.array, Hd: np.array, K: np.array, jacobian="geometric"
):
    """Using Homogeneous Transformation Matrices, this function computes Inverse Kinematics of a serial robot
    given joints positions in radians. Serial robot's kinematic parameters have to be set before using this function

    Args:
      robot (Serial): serial robot (this won't work with other type of robots)
      q0 (np.array): initial conditions of joints
      Hd (np.array): desired pose represented with an Homogeneous Transformation Matrix or Axis - Angle vector
      K (np.array): gain matrix to guarantee stability
      jacobian (str, optional): 'geometric' or 'analytic'

    Returns:
      q (np.array): joints positions to reach desired pose
    """

    # Get the shape of Hd
    r, s = Hd.shape

    # Convert desired pose to Axis - Angle Vector if this is an Homogeneous Transformation Matrix
    Xd = axisAngle(Hd) if r == 4 else Hd

    # Extract rotation matrix from desired pose and convert it to Homogeneous Transformation Matrix
    Rd = np.append(
        np.append(Hd[0:3, 0:3], np.zeros((3, 1)), axis=1),
        np.array([[0, 0, 0, 1]]),
        axis=0,
    )

    # Set initial conditions
    q = q0.reshape(q0.shape)

    # Auxiliar variable to keep original joints positions
    z = robot.jointsPositions.copy()

    # Iterator
    j = 0

    # While iterator is less than 15000 or absolute value of max
    while j < 15000:
        # Set calculated positions on robot
        robot.jointsPositions = q[:, -1].reshape(q0.shape)

        # Calculate forward kinematics
        fkH = forwardHTM(robot)

        # Set Axis - Angle vector
        X = axisAngle(fkH[-1])

        # Extract rotation matrix from current pose and convert it to Homogeneous Transformation Matrix
        R = np.append(
            np.append(fkH[-1][0:3, 0:3], np.zeros((3, 1)), axis=1),
            np.array([[0, 0, 0, 1]]),
            axis=0,
        )

        # Calculate error
        e = np.append(Xd[0:3, :] - X[0:3, :], axisAngle(Rd.dot(R.T))[3:6], axis=0)

        # If error is less equal to 0.001
        if np.abs(np.max(e)) <= 0.001:
            # Solution found!
            print("\nSolution found using Homogeneous Transformation Matrices!")
            # Stop loop
            break

        # Calculate Jacobian Matrix
        J = (
            geometricJacobian(robot)
            if jacobian == "geometric"
            else (
                analyticJacobian(robot)
                if jacobian == "analytic"
                else np.zeros((6, q0.shape[0]))
            )
        )

        # Calculate new joints positions
        solution = rungeKutta4(
            f=(np.linalg.pinv(J)).dot(K).dot(e), F=q[:, -1].reshape(q0.shape)
        )

        # Append results
        q = np.append(q, solution, axis=1)

        j += 1

    # Restore original joints positions to don't affect other calculations
    robot.jointsPositions = z.copy()

    return q


if __name__ == "__main__":

    """
    THIS SECTION IS FOR TESTING PURPOSES ONLY
    """

    print("Z")
