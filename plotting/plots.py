import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


# Keep animation objects alive until after plt.show()
_LIVE_ANIMS = []


def hold_anims(anims):
    """Store animation refs so they survive until plt.show()."""
    if anims is None:
        return
    if isinstance(anims, (list, tuple)):
        _LIVE_ANIMS.extend(anims)
    else:
        _LIVE_ANIMS.append(anims)


def plot_all(logs, time_vec, center=None, targets=None):
    T = len(time_vec)

    # ===================== TORQUES / RESIDUALS =====================
    plt.figure("Torques")
    plt.subplot(3, 1, 1)
    plt.plot(time_vec, logs.tau_des_log[:T, 0], label="τ_des_1")
    plt.plot(time_vec, logs.tau_real_log[:T, 0], label="τ_real_1")
    plt.legend()
    plt.ylabel("N·m")
    plt.title("Joint 1")

    plt.subplot(3, 1, 2)
    plt.plot(time_vec, logs.tau_des_log[:T, 1], label="τ_des_2")
    plt.plot(time_vec, logs.tau_real_log[:T, 1], label="τ_real_2")
    plt.legend()
    plt.ylabel("N·m")
    plt.title("Joint 2")

    plt.subplot(3, 1, 3)
    plt.plot(time_vec, logs.res_log[:T, 0], label="res_1")
    plt.plot(time_vec, logs.res_log[:T, 1], label="res_2")
    plt.legend()
    plt.xlabel("time [s]")
    plt.ylabel("N·m")
    plt.title("Residual")
    plt.xlim(time_vec[0], time_vec[-1])
    plt.tight_layout()

    # ===================== SATURATION & NORMS =====================
    plt.figure("Saturation & norms")
    plt.subplot(3, 1, 1)
    plt.plot(time_vec, logs.sat_min_pct[:T], label="% at min")
    plt.plot(time_vec, logs.sat_max_pct[:T], label="% at max")
    plt.legend()
    plt.ylabel("%")

    plt.subplot(3, 1, 2)
    plt.plot(time_vec, logs.act_norm[:T])
    plt.ylabel("||a||₂")

    plt.subplot(3, 1, 3)
    plt.plot(time_vec, logs.condR_log[:T], label="cond(R)")
    plt.plot(time_vec, logs.condM_log[:T], label="cond(R·Fmax)")
    plt.yscale("log")
    plt.legend()
    plt.xlabel("time [s]")
    plt.ylabel("cond")
    plt.xlim(time_vec[0], time_vec[-1])
    plt.tight_layout()

    # ===================== JOINT TRAJECTORIES =====================
    plt.figure("Joint trajectories")
    plt.subplot(2, 1, 1)
    plt.plot(time_vec, np.rad2deg(logs.q_log[:T, 0]), label="q1")
    plt.plot(time_vec, np.rad2deg(logs.qref_log[:T, 0]), "--", label="q1_ref")
    plt.legend()
    plt.ylabel("deg")
    plt.title("Shoulder")

    plt.subplot(2, 1, 2)
    plt.plot(time_vec, np.rad2deg(logs.q_log[:T, 1]), label="q2")
    plt.plot(time_vec, np.rad2deg(logs.qref_log[:T, 1]), "--", label="q2_ref")
    plt.legend()
    plt.ylabel("deg")
    plt.xlabel("time [s]")
    plt.title("Elbow")
    plt.xlim(time_vec[0], time_vec[-1])
    plt.tight_layout()

    # ===================== XY TRAJECTORY =====================
    plt.figure("XY trajectory")
    plt.plot(
        logs.xref_log[:T, 0],
        logs.xref_log[:T, 1],
        "--",
        label="ref",
    )
    plt.plot(
        logs.x_log[:T, 0],
        logs.x_log[:T, 1],
        "-",
        label="actual",
    )
    if targets is not None:
        plt.scatter(targets[:, 0], targets[:, 1], marker="x", label="targets")
    if center is not None:
        plt.scatter([center[0]], [center[1]], marker="o", label="center")
    ax_xy = plt.gca()
    ax_xy.set_aspect("equal")
    plt.legend()
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("End-effector trajectory")
    plt.tight_layout()

       # ===================== PER-MUSCLE ACTIVATIONS (3x2) =====================
    if hasattr(logs, "act_log") and logs.act_log is not None:
        act = logs.act_log[:T]           # (T, n_muscles)
        n_muscles = act.shape[1]

        plt.figure("Muscle activations (3x2)")
        rows, cols = 3, 2
        max_plots = rows * cols
        n_plots = min(n_muscles, max_plots)

        for m in range(n_plots):
            ax = plt.subplot(rows, cols, m + 1)
            y = act[:, m]

            ax.plot(time_vec, y)

            # ---- per-muscle auto-scale with a bit of padding ----
            ymin = float(np.min(y))
            ymax = float(np.max(y))
            if np.isclose(ymin, ymax):
                # Flat signal: give a small band around the value
                pad = 0.05 * (abs(ymax) + 1e-6)
                ymin -= pad
                ymax += pad
            else:
                pad = 0.05 * (ymax - ymin)
                ymin -= pad
                ymax += pad
            ax.set_ylim(ymin, ymax)
            # ------------------------------------------------------

            ax.set_title(f"Muscle m{m+1}")
            if m // cols == rows - 1:
                ax.set_xlabel("time [s]")
            ax.set_ylabel("activation")

        plt.suptitle("Per-muscle activations")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # ===================== ACTIVATION HEATMAP =====================
        plt.figure("Muscle activation heatmap")
        # act: (T, n_muscles) -> transpose to (n_muscles, T)
        im = plt.imshow(
            act.T,
            aspect="auto",
            origin="lower",
            extent=[time_vec[0], time_vec[-1], 0.5, n_muscles + 0.5],
        )
        plt.colorbar(im, label="activation")
        plt.yticks(
            np.arange(1, n_muscles + 1),
            [f"m{m+1}" for m in range(n_muscles)],
        )
        plt.xlabel("time [s]")
        plt.ylabel("muscle")
        plt.title("Muscle activations (heatmap)")
        plt.tight_layout()

        # ===================== MUSCLE CORRELATION MATRIX =====================
        plt.figure("Muscle activation correlation")
        if act.shape[0] > 1:
            corr = np.corrcoef(act.T)  # (n_muscles, n_muscles)
            im2 = plt.imshow(
                corr,
                vmin=-1.0,
                vmax=1.0,
                cmap="coolwarm",
                origin="lower",
            )
            plt.colorbar(im2, label="corr")
            ticks = np.arange(n_muscles)
            labels = [f"m{m+1}" for m in range(n_muscles)]
            plt.xticks(ticks, labels, rotation=45, ha="right")
            plt.yticks(ticks, labels)
            plt.title("Pairwise muscle activation correlation")
            plt.tight_layout()
        else:
            ax = plt.gca()
            ax.text(
                0.5,
                0.5,
                "Not enough samples for correlation",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            plt.axis("off")


def make_animations(
    logs, time_vec, env, playback=1.0, downsample=3, center=None, targets=None
):
    """
    Create animations:
      - bar plot of muscle activations
      - robot motion with full robot always visible
    """
    if len(time_vec) == 0:
        return

    T = len(time_vec)
    dt = time_vec[1] - time_vec[0] if len(time_vec) > 1 else env.dt
    idxs = np.arange(0, T, downsample)
    t_sub = time_vec[idxs]

    # ===================== ACTIVATION BARS =====================
    figA, axA = plt.subplots()
    axA.set_title("Activation bars")
    axA.set_ylim(0, 1.0)
    axA.set_ylabel("activation")
    axA.set_xlabel("time [s]")

    act0 = logs.act_log[idxs[0]]
    bars = axA.bar(range(env.n_muscles), act0)

    def update_bars(i):
        for j, b in enumerate(bars):
            b.set_height(logs.act_log[idxs[i], j])
        axA.set_xlabel(f"time = {t_sub[i]:.2f} s")
        return bars

    aniA = animation.FuncAnimation(
        figA,
        update_bars,
        frames=len(idxs),
        interval=dt * 1000 * downsample / playback,
        blit=False,
    )

    # ===================== ROBOT MOTION =====================
    figR, axR = plt.subplots()
    axR.set_aspect("equal")
    axR.set_title("Robot motion")
    axR.set_xlabel("x [m]")
    axR.set_ylabel("y [m]")

    # Targets & center
    if targets is not None:
        axR.scatter(targets[:, 0], targets[:, 1], marker="x", alpha=0.6, label="targets")
    if center is not None:
        axR.scatter([center[0]], [center[1]], marker="o", alpha=0.8, label="center")

    # Pre-compute global axis limits so robot is always fully visible
    if hasattr(logs, "links_xy") and logs.links_xy is not None:
        links_xy = np.asarray(logs.links_xy)  # (T, n_links, 2)
        xs = links_xy[..., 0]
        ys = links_xy[..., 1]
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        # Add small padding
        pad_x = 0.1 * (x_max - x_min + 1e-6)
        pad_y = 0.1 * (y_max - y_min + 1e-6)
        axR.set_xlim(x_min - pad_x, x_max + pad_x)
        axR.set_ylim(y_min - pad_y, y_max + pad_y)

    if targets is not None or center is not None:
        axR.legend()

    (line_links,) = axR.plot([], [], "-o", lw=2)
    (trail,) = axR.plot([], [], "-", alpha=0.5)
    ee_pts = []

    def update_robot(i):
        pts = logs.links_xy[idxs[i]]  # (n_links, 2)
        line_links.set_data(pts[:, 0], pts[:, 1])

        ee = pts[-1]
        ee_pts.append(ee.copy())
        trail_arr = np.array(ee_pts)
        trail.set_data(trail_arr[:, 0], trail_arr[:, 1])

        return line_links, trail

    aniR = animation.FuncAnimation(
        figR,
        update_robot,
        frames=len(idxs),
        interval=dt * 1000 * downsample / playback,
        blit=False,
    )

    # Return all animations so caller can hold them:
    #   anims = make_animations(...)
    #   hold_anims(anims)
    #   plt.show()
    return [aniA, aniR]
