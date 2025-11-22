import numpy as np


class LogBuffer:
    def __init__(self, steps, n_muscles):
        self.steps = steps
        self.n_muscles = n_muscles
        self.k = 0
        self.x_log = np.zeros((steps, 2))
        self.xd_log = np.zeros((steps, 2))
        self.xref_log = np.zeros((steps, 2))
        self.q_log = np.zeros((steps, 2))
        self.qref_log = np.zeros((steps, 2))
        self.tau_des_log = np.zeros((steps, 2))
        self.tau_real_log = np.zeros((steps, 2))
        self.res_log = np.zeros((steps, 2))
        self.act_log = np.zeros((steps, n_muscles))
        self.act_norm = np.zeros(steps)
        self.sat_min_pct = np.zeros(steps)
        self.sat_max_pct = np.zeros(steps)
        self.condR_log = np.zeros(steps)
        self.condM_log = np.zeros(steps)
        self.links_xy = np.zeros((steps, 4, 2))

    def record(self, env, diag, tau_real, qref):
        k = self.k
        x_d, xd_d, _ = diag["xref_tuple"]
        self.x_log[k] = diag["x"]
        self.xd_log[k] = diag["xd"]
        self.xref_log[k] = x_d
        self.q_log[k] = diag["q"]
        self.qref_log[k] = qref
        self.tau_des_log[k] = diag["tau_des"]
        self.tau_real_log[k] = tau_real
        #self.res_log[k] = diag["tau_des"] - tau_real
        # safer:
        tau_des = diag.get("tau_des", None)
        self.res_log[k] = (0.0 if tau_des is None else tau_des) - tau_real

        self.act_log[k] = diag["act"]
        self.act_norm[k] = np.linalg.norm(diag["act"])
        self.sat_min_pct[k] = 100.0 * np.mean(
            diag["act"] <= (env.muscle.min_activation + 1e-3)
        )
        self.sat_max_pct[k] = 100.0 * np.mean(diag["act"] >= (1.0 - 1e-3))
        R = diag["R"]
        Fmax = diag["Fmax"]
        try:
            self.condR_log[k] = np.linalg.cond(R)
        except np.linalg.LinAlgError:
            self.condR_log[k] = np.inf
        try:
            self.condM_log[k] = np.linalg.cond(R @ np.diag(Fmax))
        except np.linalg.LinAlgError:
            self.condM_log[k] = np.inf
        from model_lib.skeleton_numpy import forwardHTM_cached

        F1, F2, Fee = forwardHTM_cached(env.skeleton._robot, symbolic=False)
        base = np.array([0.0, 0.0])
        self.links_xy[k, 0] = base
        self.links_xy[k, 1] = F1[:2, 3]
        self.links_xy[k, 2] = F2[:2, 3]
        self.links_xy[k, 3] = Fee[:2, 3]
        self.k += 1

    def time(self, dt):
        return (self.k, np.arange(self.k) * dt)
