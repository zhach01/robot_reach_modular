# Plot_torch.py
# -*- coding: utf-8 -*-
"""
Torch-friendly plotting and animation utilities.

This is a Torch-aware rewrite of Plot.py:
- Accepts both torch.Tensors and NumPy arrays as inputs.
- Safely handles GPU tensors by moving data to CPU only for plotting.
- Internally uses your Torch kinematics stack:
    - forwardHTM, forwardCOMHTM from HTM_kinematics_torch
    - axisAngle from DifferentialHTM_torch

NOTE:
- These functions are *visualization utilities*; they detach tensors
  before sending data to Matplotlib. They are not meant to be part
  of a differentiable compute graph (but they do not break your
  core dynamics / control code).
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (imported for side-effect)
except Exception:
    pass

import torch
from torch import Tensor

# --- imports from your Torch kinematics stack ---
try:
    from lib.kinematics.HTM_kinematics_torch import forwardHTM, forwardCOMHTM
except Exception:
    try:
        from HTM_kinematics_torch import forwardHTM, forwardCOMHTM  # type: ignore
    except Exception:
        forwardHTM = None
        forwardCOMHTM = None

try:
    from lib.kinematics.DifferentialHTM_torch import axisAngle
except Exception:
    try:
        from DifferentialHTM_torch import axisAngle  # type: ignore
    except Exception:
        axisAngle = None


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _to_tensor(x: Any, dtype: Optional[torch.dtype] = None) -> Tensor:
    """
    Convert input (NumPy, list, Tensor) to a torch.Tensor (on CPU).

    Only for plotting, so we always put it on CPU.
    """
    if isinstance(x, Tensor):
        if x.device.type != "cpu":
            return x.detach().to("cpu")
        return x.detach()
    return torch.as_tensor(x, dtype=dtype, device="cpu")


def _to_numpy(x: Any):
    """
    Convert torch.Tensor or array-like to a NumPy array for Matplotlib.
    """
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    import numpy as np

    return np.asarray(x)


# ---------------------------------------------------------------------
# 1. Generic multi-curve plot
# ---------------------------------------------------------------------
def graph(
    function: Any,
    title: str = r"",
    labels: str = r"",
    complement: str = r"",
    xlabel: str = r"",
    ylabel: str = r"",
    save: bool = False,
    name: str = "zGraph",
    transparent: bool = False,
    GUI: bool = False,
):
    """
    Plot any given multi-curve function.

    Parameters
    ----------
    function : array-like or torch.Tensor
        Shape: (n_signals, T)
    title : str
    labels : str
        Prefix for each curve's legend (index i+1 appended).
    complement : str
        Extra string appended after index in label.
    xlabel, ylabel : str
    save : bool
        If True, saves <name>.png (overwrites existing).
    name : str
        Base filename for saving.
    transparent : bool
        Passed to plt.savefig(..., transparent=transparent).
    GUI : bool
        If True, returns the figure instead of calling plt.show().
    """
    F = _to_tensor(function)
    if F.dim() == 1:
        F = F.unsqueeze(0)  # (1,T)

    fig = plt.figure()
    plt.clf()
    plt.title(title)

    n_sig, _T = F.shape

    for i in range(n_sig):
        y = _to_numpy(F[i, :])
        plt.plot(y, label=labels + str(i + 1) + complement)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend(loc="best")

    if save:
        png_name = f"{name}.png"
        if os.path.isfile("./" + png_name):
            os.remove("./" + png_name)
        plt.savefig(png_name, transparent=transparent)

    if GUI:
        return fig
    else:
        plt.show()


# ---------------------------------------------------------------------
# 2. 1D trajectory with sample points
# ---------------------------------------------------------------------
def trajectory(
    function: Any,
    points: Any,
    steps: Any,
    title: str = r"",
    variable: str = r"",
    h: float = 0.003,
    GUI: bool = False,
):
    """
    Plot 1D trajectories with discrete points overlay.

    Parameters
    ----------
    function : array-like or torch.Tensor
        Shape: (n_traj, T)
    points : array-like or torch.Tensor
        Shape: (n_traj, T_points)
    steps : array-like or torch.Tensor
        Shape: (T_points,)
        Time steps (seconds, or same unit as h * T).
    h : float
        Base discrete time step used to scale t for x-axis.
    title : str
    variable : str
        Prefix for each trajectory label.
    GUI : bool
        If True, returns the figure instead of plt.show().
    """
    F = _to_tensor(function)
    P = _to_tensor(points)
    steps_t = _to_tensor(steps, dtype=torch.get_default_dtype())

    if F.dim() == 1:
        F = F.unsqueeze(0)
    if P.dim() == 1:
        P = P.unsqueeze(0)

    # cumulative time and x-axis for scatter
    t = torch.cumsum(steps_t, dim=-1)  # (T_points,)
    x_scatter = _to_numpy(t / h)

    fig = plt.figure()
    plt.title(title)

    n_traj, _T = F.shape

    for i in range(n_traj):
        y = _to_numpy(F[i, :])
        pts = _to_numpy(P[i, :])
        plt.plot(y, label=variable + str(i + 1) + r"$")
        plt.scatter(x=x_scatter, y=pts, c="red")

    plt.xlabel(r"Time [miliseconds]")
    plt.ylabel(r"Amplitude")
    plt.legend(loc="best")
    plt.grid()

    if GUI:
        return fig
    else:
        plt.show()


# ---------------------------------------------------------------------
# 3. 3D end-effector trajectory + orientation
# ---------------------------------------------------------------------
def trajectory3D(
    robot: Any,
    q: Any,
    m: int,
    scatter: bool = False,
    GUI: bool = False,
):
    """
    Plot robot end-effector behavior in R^3 given joint trajectories.

    Parameters
    ----------
    robot : robot object
        Must be compatible with:
            - forwardHTM(robot, m=m)
            - axisAngle(fkHTM)
        Typically your Serial-like robot with Torch DH params.
    q : array-like or torch.Tensor
        Joint trajectories, shape: (n_joints, T)
    m : int
        Number of frames (same as in forwardHTM).
    scatter : bool
        If True, scatter plot instead of line.
    GUI : bool
        If True, returns (fig_pos, fig_x, fig_y, fig_z),
        else calls plt.show().
    """
    if forwardHTM is None or axisAngle is None:
        raise RuntimeError(
            "trajectory3D requires forwardHTM and axisAngle from "
            "your Torch kinematics stack (HTM_kinematics_torch / "
            "DifferentialHTM_torch)."
        )

    q_t = _to_tensor(q, dtype=torch.get_default_dtype())
    if q_t.dim() != 2:
        raise ValueError(f"q must be 2D (n_joints, T), got shape {tuple(q_t.shape)}")

    r, s = q_t.shape  # n_joints, T

    # Figures
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    fig1 = plt.figure()
    ax1 = fig1.gca()
    fig2 = plt.figure()
    ax2 = fig2.gca()
    fig3 = plt.figure()
    ax3 = fig3.gca()

    # Position figure labels
    ax.set_title("End - effector Position", fontsize=16)
    ax.set_xlabel(r"$x$", color="red", fontsize=16)
    ax.set_ylabel(r"$y$", color="green", fontsize=16)
    ax.set_zlabel(r"$z$", color="blue", fontsize=16)

    # Orientation vs time labels
    ax1.set_title("End - effector Orientation", fontsize=16)
    ax1.set_xlabel(r"Time [milliseconds]", fontsize=14)
    ax1.set_ylabel(r"x [rad]", fontsize=14)

    ax2.set_title("End - effector Orientation", fontsize=16)
    ax2.set_xlabel(r"Time [milliseconds]", fontsize=14)
    ax2.set_ylabel(r"y [rad]", fontsize=14)

    ax3.set_title("End - effector Orientation", fontsize=16)
    ax3.set_xlabel(r"Time [milliseconds]", fontsize=14)
    ax3.set_ylabel(r"z [rad]", fontsize=14)

    # Initialize robot state at first time step
    q0 = q_t[:, 0].view(r, 1)
    # Set both q and jointsPositions for compatibility
    setattr(robot, "q", q0)
    setattr(robot, "jointsPositions", q0)

    framesHTM, fkHTM = forwardHTM(robot, m=m)
    X = axisAngle(fkHTM)  # expected shape: (6,1) or (6,T0)

    X_list = [X]

    # Propagate over time steps
    for j in range(1, s):
        qj = q_t[:, j].view(r, 1)
        setattr(robot, "q", qj)
        setattr(robot, "jointsPositions", qj)

        framesHTM, fkHTM = forwardHTM(robot, m=m)
        X_j = axisAngle(fkHTM)  # (6,1)
        X_list.append(X_j)

    # Concatenate along time
    X_cat = torch.cat(X_list, dim=1)  # (6, T)

    # Position (3D)
    X_pos = _to_numpy(X_cat[0:3, :])
    if not scatter:
        ax.plot(xs=X_pos[0, :], ys=X_pos[1, :], zs=X_pos[2, :], color="magenta")
    else:
        ax.scatter(xs=X_pos[0, :], ys=X_pos[1, :], zs=X_pos[2, :])

    # Orientation
    X_ori = _to_numpy(X_cat[3:6, :])
    ax1.plot(X_ori[0, :], color="red")
    ax1.grid()
    ax2.plot(X_ori[1, :], color="green")
    ax2.grid()
    ax3.plot(X_ori[2, :], color="blue")
    ax3.grid()

    if GUI:
        return fig, fig1, fig2, fig3
    else:
        plt.show()


# ---------------------------------------------------------------------
# 4. 3D task-space trajectory (x(t), y(t), z(t))
# ---------------------------------------------------------------------
def taskSpace(
    x: Any,
    y: Any,
    z: Any,
    scatter: bool = False,
    GUI: bool = False,
):
    """
    Plot 3D task-space trajectory.

    Parameters
    ----------
    x, y, z : array-like or torch.Tensor
        Each shape: (n_traj, T)
    scatter : bool
        If True, scatter instead of line.
    GUI : bool
        If True, returns fig instead of plt.show().
    """
    X = _to_tensor(x)
    Y = _to_tensor(y)
    Z = _to_tensor(z)

    if X.dim() == 1:
        X = X.unsqueeze(0)
        Y = Y.unsqueeze(0)
        Z = Z.unsqueeze(0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_title("Task Space Trajectory", fontsize=16)
    ax.set_xlabel(r"$x$", color="red", fontsize=16)
    ax.set_ylabel(r"$y$", color="green", fontsize=16)
    ax.set_zlabel(r"$z$", color="blue", fontsize=16)

    n_traj, _T = X.shape

    for i in range(n_traj):
        xs = _to_numpy(X[i, :])
        ys = _to_numpy(Y[i, :])
        zs = _to_numpy(Z[i, :])
        if not scatter:
            ax.plot(xs=xs, ys=ys, zs=zs, color="magenta")
        else:
            ax.scatter(xs=xs, ys=ys, zs=zs)

    plt.grid()
    if GUI:
        return fig
    else:
        plt.show()


# ---------------------------------------------------------------------
# 5. Robot animation in 3D
# ---------------------------------------------------------------------
_LIVE_ANIMS: list[FuncAnimation] = []  # keep refs alive until after plt.show()


def hold_anims(anims: Sequence[FuncAnimation]):
    """Keep animation objects alive (to avoid Matplotlib GC warnings)."""
    _LIVE_ANIMS.extend(anims)


def animation(
    robot: Any,
    q: Any,
    plotBodies: bool = True,
    plotFrames: bool = False,
    plotCOMs: bool = False,
    delayPerFrame: float | int = 1,
    repeatAnimation: bool = False,
    GUI: bool = False,
):
    """
    Animate robot behavior and attached frames / COMs in 3D.

    Parameters
    ----------
    robot : robot object
        Must be compatible with forwardHTM / forwardCOMHTM.
        Should have:
            - robot.linksLengths (list-like)
            - robot.name (str)
            - robot.q and/or robot.jointsPositions (n,1)
    q : array-like or torch.Tensor
        Joint trajectories, shape: (n_joints, T)
    plotBodies, plotFrames, plotCOMs : bool
        Whether to draw link segments, joint frames and COM frames.
    delayPerFrame : float or int
        Delay between frames in milliseconds for animation.
    repeatAnimation : bool
        Whether to repeat the animation.
    GUI : bool
        If True, returns fig instead of plt.show().
    """
    if forwardHTM is None or forwardCOMHTM is None:
        raise RuntimeError(
            "animation() requires forwardHTM and forwardCOMHTM from "
            "your Torch kinematics stack (HTM_kinematics_torch)."
        )

    q_t = _to_tensor(q)
    if q_t.dim() != 2:
        raise ValueError(f"q must be 2D (n_joints, T), got shape {tuple(q_t.shape)}")

    n, T = q_t.shape

    # Figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Scale for visualization
    links_lengths = getattr(robot, "linksLengths", None)
    if links_lengths is None:
        # fallback: assume unit links
        links_lengths = [1.0] * n

    # scaling factor 'a' as in original code
    if any(link < 1 for link in links_lengths):
        a = 10.0
    elif any(link > 100 for link in links_lengths):
        a = 0.01
    else:
        a = 1.0

    # Figure limits based on link lengths
    import math

    limit = a * math.sqrt(sum([float(link) ** 2 for link in links_lengths]))

    # number of reference frames: joints + base
    m = n + 1

    # Titles and labels
    ax.set_title(getattr(robot, "name", "Robot"), fontsize=16)
    ax.set_xlabel("$x$", color="red", fontsize=16)
    ax.set_ylabel("$y$", color="green", fontsize=16)
    ax.set_zlabel("$z$", color="blue", fontsize=16)

    # Inertial frame
    ax.plot(
        xs=[0, 1], ys=[0, 0], zs=[0, 0], color="red", linestyle="dashed", marker="o"
    )
    ax.plot(
        xs=[0, 0], ys=[0, 1], zs=[0, 0], color="green", linestyle="dashed", marker="o"
    )
    ax.plot(
        xs=[0, 0], ys=[0, 0], zs=[0, 1], color="blue", linestyle="dashed", marker="o"
    )

    def system(q_frame: Tensor | Any):
        """
        Draw a single frame of the animation for a given joint vector.
        q_frame is 1D (n,) passed by FuncAnimation from frames=q_t.T
        """
        # Clear axes (but keep labels/limits)
        ax.cla()

        # Redraw inertial axes
        ax.plot(
            xs=[0, 1],
            ys=[0, 0],
            zs=[0, 0],
            color="red",
            linestyle="dashed",
            marker="o",
        )
        ax.plot(
            xs=[0, 0],
            ys=[0, 1],
            zs=[0, 0],
            color="green",
            linestyle="dashed",
            marker="o",
        )
        ax.plot(
            xs=[0, 0],
            ys=[0, 0],
            zs=[0, 1],
            color="blue",
            linestyle="dashed",
            marker="o",
        )

        # Set limits and labels again
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
        ax.set_title(getattr(robot, "name", "Robot"), fontsize=16)
        ax.set_xlabel("$x$", color="red", fontsize=16)
        ax.set_ylabel("$y$", color="green", fontsize=16)
        ax.set_zlabel("$z$", color="blue", fontsize=16)

        q_vec = _to_tensor(q_frame).view(n, 1)  # (n,1)

        # Set robot state (q and jointsPositions)
        setattr(robot, "q", q_vec)
        setattr(robot, "jointsPositions", q_vec)

        # Kinematics for joints
        framesHTM, fkHTM = forwardHTM(robot, m=m)

        # Kinematics for COM
        framesCOMHTM, fkCOMHTM = forwardCOMHTM(robot, m=5)

        # Loop over frames 1..n
        for frame in range(1, len(framesHTM)):
            # Rigid body segments
            if plotBodies:
                ba = a * framesHTM[frame - 1][:-1, 3]
                bb = a * framesHTM[frame][:-1, 3]
                ba_np = _to_numpy(ba)
                bb_np = _to_numpy(bb)
                ax.plot(
                    xs=[ba_np[0], bb_np[0]],
                    ys=[ba_np[1], bb_np[1]],
                    zs=[ba_np[2], bb_np[2]],
                    color="brown",
                    linewidth=4.5,
                    marker="o",
                )

            # Joint frames
            if plotFrames:
                bb = framesHTM[frame][:-1, 3]
                R = framesHTM[frame][:-1, 0:3]  # 3x3
                bb_np = _to_numpy(bb)

                # x-axis
                xb = bb + R[:, 0]
                xb_np = _to_numpy(xb)
                ax.plot(
                    xs=[bb_np[0], xb_np[0]],
                    ys=[bb_np[1], xb_np[1]],
                    zs=[bb_np[2], xb_np[2]],
                    color="red",
                    linestyle="dashed",
                    marker="2",
                )

                # y-axis
                yb = bb + R[:, 1]
                yb_np = _to_numpy(yb)
                ax.plot(
                    xs=[bb_np[0], yb_np[0]],
                    ys=[bb_np[1], yb_np[1]],
                    zs=[bb_np[2], yb_np[2]],
                    color="green",
                    linestyle="dashed",
                    marker="2",
                )

                # z-axis
                zb = bb + R[:, 2]
                zb_np = _to_numpy(zb)
                ax.plot(
                    xs=[bb_np[0], zb_np[0]],
                    ys=[bb_np[1], zb_np[1]],
                    zs=[bb_np[2], zb_np[2]],
                    color="blue",
                    linestyle="dashed",
                    marker="2",
                )

            # COM frames
            if plotCOMs:
                xCOMa = a * framesCOMHTM[frame][:-1, 3]
                R = framesHTM[frame][:-1, 0:3]

                xCOMa_np = _to_numpy(xCOMa)

                # x-axis at COM
                xCOMb = xCOMa + R[:, 0]
                xCOMb_np = _to_numpy(xCOMb)
                ax.plot(
                    xs=[xCOMa_np[0], xCOMb_np[0]],
                    ys=[xCOMa_np[1], xCOMb_np[1]],
                    zs=[xCOMa_np[2], xCOMb_np[2]],
                    color="red",
                    linestyle=":",
                )

                # y-axis at COM
                yCOMb = xCOMa + R[:, 1]
                yCOMb_np = _to_numpy(yCOMb)
                ax.plot(
                    xs=[xCOMa_np[0], yCOMb_np[0]],
                    ys=[xCOMa_np[1], yCOMb_np[1]],
                    zs=[xCOMa_np[2], yCOMb_np[2]],
                    color="green",
                    linestyle=":",
                )

                # z-axis at COM
                zCOMb = xCOMa + R[:, 2]
                zCOMb_np = _to_numpy(zCOMb)
                ax.plot(
                    xs=[xCOMa_np[0], zCOMb_np[0]],
                    ys=[xCOMa_np[1], zCOMb_np[1]],
                    zs=[xCOMa_np[2], zCOMb_np[2]],
                    color="blue",
                    linestyle=":",
                )

    # Create animation
    ani = FuncAnimation(
        fig,
        system,
        frames=q_t.T,  # (T, n) -> each frame is 1D (n,)
        interval=delayPerFrame,
        repeat=repeatAnimation,
    )
    hold_anims([ani])

    if GUI:
        # Draw first frame once and return fig (for embedding in GUI)
        system(q_t[:, 0])
        return fig
    else:
        plt.show()


# ---------------------------------------------------------------------
# Simple smoke test: basic curves (no robot dependency)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("[Plot_torch] Simple smoke test (graph / trajectory / taskSpace) ...")

    T = 100
    t = torch.linspace(0.0, 1.0, T)

    # graph: 2 curves
    y = torch.stack([torch.sin(2 * 3.1415 * t), torch.cos(2 * 3.1415 * t)], dim=0)
    graph(y, title="graph() smoke test", labels="y_", complement="", xlabel="k", ylabel="amp")

    # trajectory: 1 curve with sample points each 10 steps
    steps = torch.ones(10) * 0.003
    points = y[0, ::10][:10].unsqueeze(0)
    trajectory(y[0:1, :], points, steps, title="trajectory() smoke test", variable="q")

    # taskSpace: 1 circle in xy-plane
    x = torch.cos(2 * 3.1415 * t).unsqueeze(0)
    y2 = torch.sin(2 * 3.1415 * t).unsqueeze(0)
    z = torch.zeros_like(x)
    taskSpace(x, y2, z, scatter=False)

    print("[Plot_torch] Smoke test done.")
