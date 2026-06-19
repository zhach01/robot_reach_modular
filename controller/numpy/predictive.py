# controller/numpy/predictive.py
# Feedback delay compensation as part of the control law (not the simulator).
#
# A controller that commands an actuator through a known transport delay `delay_s`
# acts on stale feedback: the excitation it issues now only reaches the muscles
# `delay_s` later, by which time the arm has moved on. Classical fix (O. Smith,
# 1957): give the controller the state *predicted one delay ahead*, so the command
# it produces is the correct one for the instant it actually lands.
#
# This wraps ANY of the four controllers (impedance / passivity / sliding / OSC)
# without touching them: they read the effector state dict, so the wrapper swaps in
# the predicted state for the duration of compute() and restores the true state
# before the plant integrates. The compensation therefore belongs to the controller,
# not the benchmark harness.
#
# Predictor: second-order Taylor lead from the joint velocity derivative, mapped to
# Cartesian through the effector's own forward kinematics / Jacobian (joint2cartesian).
# Acceleration is estimated from the clean internal velocity (a one-step difference of
# the *noisy* measurement would be amplified by 1/dt); sensor noise is then added to
# the predicted measurement, i.e. the wrapper models a controller front-end =
# state estimate (idealized observer) -> lead predictor -> noisy measurement.
import numpy as np


class PredictiveController:
    """Smith-style lead-predictor wrapper around a state-feedback controller.

    Parameters
    ----------
    inner : controller with .reset(q0) and .compute(x_d, xd_d, xdd_d) -> diag
    effector : the plant effector (provides .states and .joint2cartesian / .skeleton)
    delay_s : transport delay to compensate (s); 0 disables the lead (noise only)
    dt : control timestep (s)
    rng : optional np.random.Generator for sensor noise on the predicted measurement
    sigma_theta, sigma_x : joint-angle (rad) and endpoint (m) sensor-noise std
    """

    def __init__(self, inner, effector, delay_s, dt, rng=None,
                 sigma_theta=0.0, sigma_x=0.0):
        self.inner = inner
        self.eff = effector
        self.h = float(delay_s)
        self.dt = float(dt)
        self.rng = rng
        self.sth = float(sigma_theta)
        self.sx = float(sigma_x)
        self.qd_prev = None

    def reset(self, q0):
        self.qd_prev = None
        return self.inner.reset(q0)

    # expose the wrapped controller's attributes (diagnostics, params, ...)
    def __getattr__(self, name):
        return getattr(self.__dict__["inner"], name)

    def compute(self, x_d, xd_d, xdd_d):
        eff = self.eff
        q_true = eff.states["joint"].copy()
        x_true = eff.states["cartesian"].copy()
        q = q_true[0, :2]
        qd = q_true[0, 2:]

        # acceleration from the clean velocity derivative (avoids 1/dt noise blow-up)
        qdd = np.zeros(2) if self.qd_prev is None else (qd - self.qd_prev) / self.dt
        self.qd_prev = qd.copy()

        # second-order lead to the instant the command lands (t + delay)
        h = self.h
        q_p = q + qd * h + 0.5 * qdd * h * h
        qd_p = qd + qdd * h
        jp = np.concatenate([q_p, qd_p])[None, :]
        cp = np.asarray(eff.joint2cartesian(joint_state=jp)).copy()
        jp = jp.copy()

        # sensor noise on the predicted measurement
        if self.rng is not None:
            if self.sth:
                jp[:, :2] += self.rng.normal(0.0, self.sth, size=(1, 2))
            if self.sx:
                cp[:, :2] += self.rng.normal(0.0, self.sx, size=(1, 2))

        eff.states["joint"] = jp
        eff.states["cartesian"] = cp
        try:
            diag = self.inner.compute(x_d, xd_d, xdd_d)
        finally:
            eff.states["joint"] = q_true            # plant integrates the true state
            eff.states["cartesian"] = x_true
            sk = getattr(eff, "skeleton", None)
            if sk is not None and hasattr(sk, "_set_state"):
                sk._set_state(q_true[0, :2], q_true[0, 2:])
        return diag
