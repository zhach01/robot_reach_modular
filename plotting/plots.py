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


def _to_np(x):
    """
    Convert Torch / list / anything array-like to a NumPy array (on CPU).
    If x is already a NumPy array, returns it unchanged.
    """
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    # Torch tensor?
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    # Fallback: let NumPy try
    return np.asarray(x)


def plot_all(logs, time_vec, center=None, targets=None):
    # Ensure time_vec is NumPy 1D float
    time_vec = _to_np(time_vec).astype(float).ravel()

    # Convert all logged arrays to NumPy once
    tau_des_log = _to_np(logs.tau_des_log)
    tau_real_log = _to_np(logs.tau_real_log)
    res_log = _to_np(logs.res_log)
    sat_min_pct = _to_np(logs.sat_min_pct)
    sat_max_pct = _to_np(logs.sat_max_pct)
    act_norm = _to_np(logs.act_norm)
    condR_log = _to_np(logs.condR_log)
    condM_log = _to_np(logs.condM_log)
    q_log = _to_np(logs.q_log)
    qref_log = _to_np(logs.qref_log)
    xref_log = _to_np(logs.xref_log)
    x_log = _to_np(logs.x_log)
    act_log = _to_np(getattr(logs, "act_log", None))

    # Effective horizon = min(len(time_vec), length of tau_des_log)
    T = min(len(time_vec), tau_des_log.shape[0])
    t = time_vec[:T]

    # ===================== TORQUES / RESIDUALS =====================
    plt.figure("Torques")
    plt.subplot(3, 1, 1)
    plt.plot(t, tau_des_log[:T, 0], label="τ_des_1")
    plt.plot(t, tau_real_log[:T, 0], label="τ_real_1")
    plt.legend()
    plt.ylabel("N·m")
    plt.title("Joint 1")

    plt.subplot(3, 1, 2)
    plt.plot(t, tau_des_log[:T, 1], label="τ_des_2")
    plt.plot(t, tau_real_log[:T, 1], label="τ_real_2")
    plt.legend()
    plt.ylabel("N·m")
    plt.title("Joint 2")

    plt.subplot(3, 1, 3)
    plt.plot(t, res_log[:T, 0], label="res_1")
    plt.plot(t, res_log[:T, 1], label="res_2")
    plt.legend()
    plt.xlabel("time [s]")
    plt.ylabel("N·m")
    plt.title("Residual")
    plt.xlim(t[0], t[-1])
    plt.tight_layout()

    # ===================== SATURATION & NORMS =====================
    plt.figure("Saturation & norms")
    plt.subplot(3, 1, 1)
    plt.plot(t, sat_min_pct[:T], label="% at min")
    plt.plot(t, sat_max_pct[:T], label="% at max")
    plt.legend()
    plt.ylabel("%")

    plt.subplot(3, 1, 2)
    plt.plot(t, act_norm[:T])
    plt.ylabel("||a||₂")

    plt.subplot(3, 1, 3)
    plt.plot(t, condR_log[:T], label="cond(R)")
    plt.plot(t, condM_log[:T], label="cond(R·Fmax)")
    plt.yscale("log")
    plt.legend()
    plt.xlabel("time [s]")
    plt.ylabel("cond")
    plt.xlim(t[0], t[-1])
    plt.tight_layout()

    # ===================== JOINT TRAJECTORIES =====================
    plt.figure("Joint trajectories")
    plt.subplot(2, 1, 1)
    plt.plot(t, np.rad2deg(q_log[:T, 0]), label="q1")
    plt.plot(t, np.rad2deg(qref_log[:T, 0]), "--", label="q1_ref")
    plt.legend()
    plt.ylabel("deg")
    plt.title("Shoulder")

    plt.subplot(2, 1, 2)
    plt.plot(t, np.rad2deg(q_log[:T, 1]), label="q2")
    plt.plot(t, np.rad2deg(qref_log[:T, 1]), "--", label="q2_ref")
    plt.legend()
    plt.ylabel("deg")
    plt.xlabel("time [s]")
    plt.title("Elbow")
    plt.xlim(t[0], t[-1])
    plt.tight_layout()


    # ===================== XY TRAJECTORY =====================
    # Square figure + equal scale, with automatic padding so it doesn't look squished
    fig_xy = plt.figure("XY trajectory", figsize=(5, 5))

    # reference and actual trajectory in workspace
    plt.plot(
        xref_log[:T, 0],
        xref_log[:T, 1],
        "--",
        label="ref",
    )
    plt.plot(
        x_log[:T, 0],
        x_log[:T, 1],
        "-",
        label="actual",
    )

    # collect all XY points to set nice bounds
    x_data = [xref_log[:T, 0], x_log[:T, 0]]
    y_data = [xref_log[:T, 1], x_log[:T, 1]]

    if targets is not None:
        targets_np = _to_np(targets)
        plt.scatter(targets_np[:, 0], targets_np[:, 1], marker="x", label="targets")
        x_data.append(targets_np[:, 0])
        y_data.append(targets_np[:, 1])

    if center is not None:
        center_np = _to_np(center)
        plt.scatter([center_np[0]], [center_np[1]], marker="o", label="center")
        x_data.append(np.array([center_np[0]]))
        y_data.append(np.array([center_np[1]]))

    # flatten + compute square bounds
    x_all = np.concatenate([np.asarray(a).ravel() for a in x_data])
    y_all = np.concatenate([np.asarray(a).ravel() for a in y_data])

    x_min, x_max = float(x_all.min()), float(x_all.max())
    y_min, y_max = float(y_all.min()), float(y_all.max())

    dx = x_max - x_min
    dy = y_max - y_min
    span = max(dx, dy, 1e-6)
    pad = 0.05 * span

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)

    ax_xy = plt.gca()
    ax_xy.set_xlim(cx - span / 2 - pad, cx + span / 2 + pad)
    ax_xy.set_ylim(cy - span / 2 - pad, cy + span / 2 + pad)
    ax_xy.set_aspect("equal", adjustable="box")

    plt.legend()
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("End-effector trajectory")
    fig_xy.tight_layout()
    # ===================== END-EFFECTOR SPEED =====================




    # ===================== PER-MUSCLE ACTIVATIONS (3x2) =====================
    if act_log is not None:
        act = act_log[:T]           # (T, n_muscles)
        n_muscles = act.shape[1]

        plt.figure("Muscle activations (3x2)")
        rows, cols = 3, 2
        max_plots = rows * cols
        n_plots = min(n_muscles, max_plots)

        for m in range(n_plots):
            ax = plt.subplot(rows, cols, m + 1)
            y = act[:, m]

            ax.plot(t, y)

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
            extent=[t[0], t[-1], 0.5, n_muscles + 0.5],
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
    # Convert time_vec to NumPy
    time_vec = _to_np(time_vec).astype(float).ravel()

    if len(time_vec) == 0:
        return

    # Convert logged arrays we need
    act_log = _to_np(getattr(logs, "act_log", None))
    links_xy = _to_np(getattr(logs, "links_xy", None))

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

    if act_log is None:
        # No activations to animate
        bars = []
    else:
        act0 = act_log[idxs[0]]
        act0 = np.asarray(act0).ravel()
        n_muscles = act0.shape[0]
        bars = axA.bar(range(n_muscles), act0)

    def update_bars(i):
        if act_log is not None:
            y = act_log[idxs[i]]
            y = np.asarray(y).ravel()
            for j, b in enumerate(bars):
                if j < len(y):
                    b.set_height(y[j])
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
        targets_np = _to_np(targets)
        axR.scatter(
            targets_np[:, 0],
            targets_np[:, 1],
            marker="x",
            alpha=0.6,
            label="targets",
        )
    if center is not None:
        center_np = _to_np(center)
        axR.scatter(
            [center_np[0]],
            [center_np[1]],
            marker="o",
            alpha=0.8,
            label="center",
        )

    # Pre-compute global axis limits so robot is always fully visible
    if links_xy is not None:
        links_xy_np = np.asarray(links_xy)
        xs = links_xy_np[..., 0]
        ys = links_xy_np[..., 1]
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
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
        if links_xy is None:
            return line_links, trail

        pts = np.asarray(links_xy)[idxs[i]]  # (n_links, 2)
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
