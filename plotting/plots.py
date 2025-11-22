import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


# Add this at top-level in this file
_LIVE_ANIMS = []  # keeps animation objects alive until after plt.show()


def hold_anims(anims):
    """Store animation refs so they survive until plt.show()."""
    _LIVE_ANIMS.extend(anims)


def plot_all(logs, time_vec, center=None, targets=None):
    plt.figure("Torques")
    plt.subplot(3, 1, 1)
    plt.plot(time_vec, logs.tau_des_log[: len(time_vec), 0], label="τ_des_1")
    plt.plot(time_vec, logs.tau_real_log[: len(time_vec), 0], label="τ_real_1")
    plt.legend()
    plt.ylabel("N·m")
    plt.title("Joint 1")

    plt.subplot(3, 1, 2)
    plt.plot(time_vec, logs.tau_des_log[: len(time_vec), 1], label="τ_des_2")
    plt.plot(time_vec, logs.tau_real_log[: len(time_vec), 1], label="τ_real_2")
    plt.legend()
    plt.ylabel("N·m")
    plt.title("Joint 2")

    plt.subplot(3, 1, 3)
    plt.plot(time_vec, logs.res_log[: len(time_vec), 0], label="res_1")
    plt.plot(time_vec, logs.res_log[: len(time_vec), 1], label="res_2")
    plt.legend()
    plt.xlabel("time [s]")
    plt.ylabel("N·m")
    plt.title("Residual")

    plt.figure("Saturation & norms")
    plt.subplot(3, 1, 1)
    plt.plot(time_vec, logs.sat_min_pct[: len(time_vec)], label="% at min")
    plt.plot(time_vec, logs.sat_max_pct[: len(time_vec)], label="% at max")
    plt.legend()
    plt.ylabel("%")
    plt.subplot(3, 1, 2)
    plt.plot(time_vec, logs.act_norm[: len(time_vec)])
    plt.ylabel("||a||₂")
    plt.subplot(3, 1, 3)
    plt.plot(time_vec, logs.condR_log[: len(time_vec)], label="cond(R)")
    plt.plot(time_vec, logs.condM_log[: len(time_vec)], label="cond(R·Fmax)")
    plt.yscale("log")
    plt.legend()
    plt.xlabel("time [s]")
    plt.ylabel("cond")

    plt.figure("Joint trajectories")
    plt.subplot(2, 1, 1)
    plt.plot(time_vec, np.rad2deg(logs.q_log[: len(time_vec), 0]), label="q1")
    plt.plot(
        time_vec, np.rad2deg(logs.qref_log[: len(time_vec), 0]), "--", label="q1_ref"
    )
    plt.legend()
    plt.ylabel("deg")
    plt.title("Shoulder")
    plt.subplot(2, 1, 2)
    plt.plot(time_vec, np.rad2deg(logs.q_log[: len(time_vec), 1]), label="q2")
    plt.plot(
        time_vec, np.rad2deg(logs.qref_log[: len(time_vec), 1]), "--", label="q2_ref"
    )
    plt.legend()
    plt.ylabel("deg")
    plt.xlabel("time [s]")
    plt.title("Elbow")

    plt.figure("XY trajectory")
    plt.plot(
        logs.xref_log[: len(time_vec), 0],
        logs.xref_log[: len(time_vec), 1],
        "--",
        label="ref",
    )
    plt.plot(
        logs.x_log[: len(time_vec), 0],
        logs.x_log[: len(time_vec), 1],
        "-",
        label="actual",
    )
    if targets is not None:
        plt.scatter(targets[:, 0], targets[:, 1], marker="x", label="targets")
    if center is not None:
        plt.scatter([center[0]], [center[1]], marker="o", label="center")
    plt.gca().set_aspect("equal")
    plt.legend()
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")


def make_animations(
    logs, time_vec, env, playback=1.0, downsample=3, center=None, targets=None
):
    if len(time_vec) == 0:
        return
    dt = time_vec[1] - time_vec[0] if len(time_vec) > 1 else env.dt
    idxs = np.arange(0, len(time_vec), downsample)
    t_sub = time_vec[idxs]

    figA, axA = plt.subplots()
    axA.set_title("Activation bars")
    axA.set_ylim(0, 1.0)
    bars = axA.bar(range(env.n_muscles), logs.act_log[idxs[0]])

    def update_bars(i):
        for j, b in enumerate(bars):
            b.set_height(logs.act_log[idxs[i], j])
        axA.set_xlabel(f"time = {t_sub[i]:.2f}s")
        return bars

    aniA = animation.FuncAnimation(
        figA,
        update_bars,
        frames=len(idxs),
        interval=dt * 1000 * downsample / playback,
        blit=False,
    )

    figR, axR = plt.subplots()
    axR.set_aspect("equal")
    axR.set_title("Robot motion")
    if targets is not None:
        axR.scatter(targets[:, 0], targets[:, 1], marker="x", alpha=0.6)
    if center is not None:
        axR.scatter([center[0]], [center[1]], marker="o", alpha=0.8)
    (line_links,) = axR.plot([], [], "-o", lw=2)
    (trail,) = axR.plot([], [], "-", alpha=0.5)
    ee_pts = []

    def update_robot(i):
        pts = logs.links_xy[idxs[i]]
        line_links.set_data(pts[:, 0], pts[:, 1])
        ee = pts[-1]
        ee_pts.append(ee.copy())
        trail.set_data(np.array(ee_pts)[:, 0], np.array(ee_pts)[:, 1])
        return line_links, trail

    aniR = animation.FuncAnimation(
        figR,
        update_robot,
        frames=len(idxs),
        interval=dt * 1000 * downsample / playback,
        blit=False,
    )
    # IMPORTANT: return ALL animations you create
    return [aniA, aniR]  # , aniHM]
