# -*- coding: utf-8 -*-
"""
Two-DOF planar arm built purely on your robotics library (HTM formulation).

Uses:
- lib.Robot.Serial for the model container
- lib.kinematics.HTM.{forwardHTM, geometricJacobian}
- lib.dynamics.DynamicsHTM.{inertiaMatrixCOM, centrifugalCoriolisCOM, gravitationalCOM}

No torch. Batching supported on inputs to ode/integrate/joint2cartesian/path2cartesian.

This version adds:
- Persistent, quantized caching of heavy library calls (D, C, g, J, FK frames)
- Stable linear solve (np.linalg.solve) for qdd instead of explicit matrix inverse
"""

from dataclasses import dataclass
from typing import Tuple, Union
import os, atexit, pickle
import numpy as np

from lib.Robot import Serial
import lib.kinematics.HTM as _htm
import lib.dynamics.DynamicsHTM as _dyn

# Try to locate the correct gravity function name once
_HAS_GRAVITATIONAL = hasattr(_dyn, "gravitationalCOM")


def _gravity_fun(robot, g, symbolic=False):
    if _HAS_GRAVITATIONAL:
        return _dyn.gravitationalCOM(robot, g=g, symbolic=symbolic)
    else:
        # Some versions export gravityCOM
        return _dyn.gravityCOM(robot, g=g, symbolic=symbolic)


# -----------------------------------------------------------------------------
# Persistent CACHE for heavy HTM/dynamics calls
# -----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
USE_CACHE = True
CACHE_DECIMALS = 4  # round q, qd to this many decimals for cache keys
CACHE_LIMIT = 20000  # cap per map to avoid unbounded growth
CACHE_PATH = os.path.join(HERE, "htm_cache.pkl")

# Cache maps
#   "M"   : inertia matrix D(q)
#   "C"   : Coriolis/centrifugal vector C(q, qd)
#   "g"   : gravity vector g(q)
#   "J"   : geometric Jacobian J(q)
#   "Jdot": time derivative of geometric Jacobian dJ/dt(q, qd)    # --- CHANGE: added
#   "F"   : forward HTM frames list (we cache link1, link2, ee only, as np.ndarray)
if USE_CACHE:
    try:
        with open(CACHE_PATH, "rb") as f:
            _CACHE = pickle.load(f)

            # --- CHANGE: schema upgrade so old pickles get the new 'Jdot' bucket ---
            def _ensure_cache_schema(cache):
                for k in ("M", "C", "g", "J", "Jdot", "F"):
                    cache.setdefault(k, {})
                return cache

            _CACHE = _ensure_cache_schema(_CACHE)

    # --- END CHANGE ---
    except Exception:
        _CACHE = {"M": {}, "C": {}, "g": {}, "J": {}, "Jdot": {}, "F": {}}  # --- CHANGE
else:
    _CACHE = {"M": {}, "C": {}, "g": {}, "J": {}, "Jdot": {}, "F": {}}  # --- CHANGE


def _key(arrs, dec=CACHE_DECIMALS):
    flat = np.concatenate([np.round(np.asarray(a).ravel(), dec) for a in arrs])
    return flat.tobytes()


def _cap(d):
    if len(d) > CACHE_LIMIT:
        # remove ~10% oldest items (unordered simple drop)
        for _ in range(max(1, len(d) // 10)):
            d.pop(next(iter(d)))
    return d


def inertiaMatrixCOM_cached(robot, symbolic=False):
    if not USE_CACHE:
        return np.asarray(_dyn.inertiaMatrixCOM(robot, symbolic=symbolic), dtype=float)
    k = _key([robot.jointsPositions[:, 0]])
    D = _CACHE["M"].get(k, None)
    if D is None:
        D = np.asarray(_dyn.inertiaMatrixCOM(robot, symbolic=False), dtype=float)
        _CACHE["M"][k] = D
        _CACHE["M"] = _cap(_CACHE["M"])
    return D


def centrifugalCoriolisCOM_cached(robot, symbolic=False):
    if not USE_CACHE:
        return np.asarray(
            _dyn.centrifugalCoriolisCOM(robot, symbolic=symbolic), dtype=float
        )
    k = _key([robot.jointsPositions[:, 0], robot.jointsVelocities[:, 0]])
    C = _CACHE["C"].get(k, None)
    if C is None:
        C = np.asarray(_dyn.centrifugalCoriolisCOM(robot, symbolic=False), dtype=float)
        _CACHE["C"][k] = C
        _CACHE["C"] = _cap(_CACHE["C"])
    return C


def gravityCOM_cached(robot, g_vec, symbolic=False):
    if not USE_CACHE:
        return np.asarray(_gravity_fun(robot, g=g_vec, symbolic=symbolic), dtype=float)
    k = _key([robot.jointsPositions[:, 0], g_vec.ravel()])
    gv = _CACHE["g"].get(k, None)
    if gv is None:
        gv = np.asarray(_gravity_fun(robot, g=g_vec, symbolic=False), dtype=float)
        _CACHE["g"][k] = gv
        _CACHE["g"] = _cap(_CACHE["g"])
    return gv


def geometricJacobian_cached(robot, symbolic=False):
    if not USE_CACHE:
        return np.asarray(_htm.geometricJacobian(robot, symbolic=symbolic), dtype=float)
    k = _key([robot.jointsPositions[:, 0]])
    J = _CACHE["J"].get(k, None)
    if J is None:
        J = np.asarray(_htm.geometricJacobian(robot, symbolic=False), dtype=float)
        _CACHE["J"][k] = J
        _CACHE["J"] = _cap(_CACHE["J"])
    return J


def forwardHTM_cached(robot, symbolic=False):
    """
    Cache only the small set we use repeatedly: link1, link2, ee frames as arrays.
    We reconstruct them as a tuple of arrays to avoid pickling library objects.
    """
    if not USE_CACHE:
        frames = _htm.forwardHTM(robot, symbolic=symbolic)
        return [np.asarray(fr, dtype=float) for fr in frames]

    k = _key([robot.jointsPositions[:, 0]])
    F = _CACHE["F"].get(k, None)
    if F is None:
        frames = _htm.forwardHTM(robot, symbolic=False)
        # store numpy arrays (link1 index 1, link2 index 2, ee is last)
        arr_link1 = np.asarray(frames[1], dtype=float)
        arr_link2 = np.asarray(frames[2], dtype=float)
        arr_ee = np.asarray(frames[-1], dtype=float)
        F = (arr_link1, arr_link2, arr_ee)
        _CACHE["F"][k] = F
        _CACHE["F"] = _cap(_CACHE["F"])
    return F  # (link1, link2, ee) arrays


# --- CHANGE: new function ---
# --- CHANGE: robust, tolerant Jdot cache wrapper (handles old pickles too) ---
def geometricJacobianDot_cached(robot, symbolic=False):
    """
    Cached wrapper around lib.kinematics.HTM.geometricJacobianDerivative(robot).

    Notes
    -----
    * dJ/dt depends on (q, qd), so the cache key includes both.
    * If running with an old cache file, we create the 'Jdot' bucket on the fly.
    * If the library doesn't expose the derivative, we raise a clear error.
      (You could add a finite-difference fallback here if needed.)
    """
    # Ensure the function exists in the library
    if not hasattr(_htm, "geometricJacobianDerivative"):
        raise AttributeError(
            "lib.kinematics.HTM.geometricJacobianDerivative(...) not found. "
            "Update lib.zip or add a finite-difference fallback."
        )

    if not USE_CACHE:
        return np.asarray(
            _htm.geometricJacobianDerivative(robot, symbolic=symbolic), dtype=float
        )

    # Key on both q and qd
    k = _key(
        [robot.jointsPositions[:, 0], robot.jointsVelocities[:, 0]], dec=CACHE_DECIMALS
    )

    # Be tolerant if 'Jdot' was missing in an older cache pickle
    jdot_bucket = _CACHE.setdefault("Jdot", {})
    dJ = jdot_bucket.get(k, None)
    if dJ is None:
        dJ = np.asarray(
            _htm.geometricJacobianDerivative(robot, symbolic=False), dtype=float
        )  # (6, n)
        jdot_bucket[k] = dJ
        _CACHE["Jdot"] = _cap(jdot_bucket)
    return dJ


# --- END CHANGE ---


def _save_cache():
    if not USE_CACHE:
        return
    try:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(_CACHE, f, protocol=4)
        print(
            f"[cache] saved {sum(len(v) for v in _CACHE.values())} entries to {CACHE_PATH}"
        )
    except Exception as e:
        print("[cache] save failed:", e)


atexit.register(_save_cache)

# ---------------------------------------------------------------------
# Base class (no torch)
# ---------------------------------------------------------------------


@dataclass
class Skeleton:
    dof: int
    space_dim: int
    name: str = "skeleton"
    pos_lower_bound: Union[float, Tuple[float, ...]] = -1.0
    pos_upper_bound: Union[float, Tuple[float, ...]] = +1.0
    vel_lower_bound: Union[float, Tuple[float, ...]] = -1000.0
    vel_upper_bound: Union[float, Tuple[float, ...]] = +1000.0
    dt: float = 0.001  # default time step
    input_dim: int = None
    state_dim: int = None
    output_dim: int = None

    def __post_init__(self):
        self.input_dim = self.input_dim or self.dof
        self.state_dim = self.state_dim or (2 * self.dof)
        self.output_dim = self.output_dim or self.state_dim

        # Ensure (1, dof) broadcastable bounds like the original
        self.pos_lower_bound = np.array(self.pos_lower_bound, dtype=float).reshape(
            1, -1
        )
        self.pos_upper_bound = np.array(self.pos_upper_bound, dtype=float).reshape(
            1, -1
        )
        self.vel_lower_bound = np.array(self.vel_lower_bound, dtype=float).reshape(
            1, -1
        )
        self.vel_upper_bound = np.array(self.vel_upper_bound, dtype=float).reshape(
            1, -1
        )

    def build(
        self,
        timestep: float,
        pos_upper_bound,
        pos_lower_bound,
        vel_upper_bound,
        vel_lower_bound,
    ):
        self.dt = float(timestep)
        self.pos_upper_bound = np.array(pos_upper_bound, dtype=float).reshape(1, -1)
        self.pos_lower_bound = np.array(pos_lower_bound, dtype=float).reshape(1, -1)
        self.vel_upper_bound = np.array(vel_upper_bound, dtype=float).reshape(1, -1)
        self.vel_lower_bound = np.array(vel_lower_bound, dtype=float).reshape(1, -1)

    @staticmethod
    def _clip(x, lb, ub):
        return np.minimum(np.maximum(x, lb), ub)

    def clip_position(self, q):
        return self._clip(q, self.pos_lower_bound, self.pos_upper_bound)

    def clip_velocity(self, pos, vel):
        # 1) clip to vel bounds
        vel = self._clip(vel, self.vel_lower_bound, self.vel_upper_bound)
        # 2) if at/over joint limits and velocity pushes farther out, zero it
        vel = np.where((vel < 0) & (pos <= self.pos_lower_bound), 0.0, vel)
        vel = np.where((vel > 0) & (pos >= self.pos_upper_bound), 0.0, vel)
        return vel

    def ode(
        self, inputs: np.ndarray, joint_state: np.ndarray, endpoint_load: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    def integrate(
        self, dt: float, state_derivative: np.ndarray, joint_state: np.ndarray
    ) -> np.ndarray:
        """
        Semi-implicit Euler (matches original behavior):
            qd_{k+1} = qd_k + qdd * dt
            q_{k+1}  = q_k  + qd_k * dt
        """
        q, qd = np.split(joint_state, 2, axis=1)
        qdd = state_derivative

        new_qd = qd + qdd * dt
        new_q = q + qd * dt

        new_qd = self.clip_velocity(new_q, new_qd)
        new_q = self.clip_position(new_q)
        return np.concatenate([new_q, new_qd], axis=1)

    def joint2cartesian(self, joint_state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def path2cartesian(
        self,
        path_coordinates: np.ndarray,  # (batch, 2, n_points)
        path_fixation_body: np.ndarray,  # (1, n_points) with {0: world, 1: link1, 2: link2}
        joint_state: np.ndarray,  # (batch, 2*dof)
    ):
        raise NotImplementedError

    @staticmethod
    def _as_batch(x, d):
        """Ensure x is 2D batch of width d."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != d:
            raise ValueError(f"Expected shape (*, {d}), got {x.shape}")
        return x


# ---------------------------------------------------------------------
# Two-DOF planar arm (HTM-based)
# ---------------------------------------------------------------------


class TwoDofArm(Skeleton):
    """
    A two-DOF planar arm powered by your library.

    Dynamics:
        qdd = D(q)^{-1} [ τ + J(q)^T f - C(q,qd) qd - g(q) ]
    where:
        - D, C, g come from lib.dynamics.DynamicsHTM
        - J is lib.kinematics.HTM.geometricJacobian (linear x-y rows)
    """

    def __init__(
        self,
        name: str = "two_dof_arm",
        m1: float = 1.864572,
        m2: float = 1.534315,
        l1g: float = 0.180496,
        l2g: float = 0.181479,
        i1: float = 0.013193,
        i2: float = 0.020062,
        l1: float = 0.309,
        l2: float = 0.26,
        viscosity: float = 0.0,
        gravity_vec=np.array(
            [0.0, 0.0, -9.81]
        ),  # R^3 gravity; -z is perpendicular to planar xy
        **kwargs,
    ):
        # Joint limits (radians), same spirit as your original code
        sho_limit = np.deg2rad([0.0, 140.0])
        elb_limit = np.deg2rad([0.0, 160.0])
        lb = [sho_limit[0], elb_limit[0]]
        ub = [sho_limit[1], elb_limit[1]]

        super().__init__(
            dof=2,
            space_dim=2,
            name=name,
            pos_lower_bound=lb,
            pos_upper_bound=ub,
            **kwargs,
        )

        # Physical params
        self.m1, self.m2 = float(m1), float(m2)
        self.L1g, self.L2g = float(l1g), float(l2g)
        self.I1, self.I2 = float(i1), float(i2)
        self.L1, self.L2 = float(l1), float(l2)
        self.c_viscosity = float(viscosity)
        self._gravity_vec = np.asarray(gravity_vec, dtype=float).reshape(3, 1)

        # Create underlying library robot (Serial). It expects column vectors (n x 1).
        self._robot = Serial(
            jointsPositions=np.zeros((2, 1), dtype=float),
            jointsVelocities=np.zeros((2, 1), dtype=float),
            jointsAccelerations=np.zeros((2, 1), dtype=float),
            linksLengths=[self.L1, self.L2],
            COMs=[self.L1g, self.L2g],
            mass=[self.m1, self.m2],
            # Simple 3x3 inertias (planar: Izz dominant; Ix,Iy tiny to avoid singularities)
            inertia=[
                np.diag([1e-6, 1e-6, self.I1]),
                np.diag([1e-6, 1e-6, self.I2]),
            ],
        )

    # --- helpers ---------------------------------------------------

    def _set_state(self, q: np.ndarray, qd: np.ndarray):
        """Push q, qd into the underlying lib robot as column vectors (n x 1)."""
        self._robot.jointsPositions = np.asarray(q, dtype=float).reshape(-1, 1)
        self._robot.jointsVelocities = np.asarray(qd, dtype=float).reshape(-1, 1)

    # --- core API --------------------------------------------------

    def ode(
        self, inputs: np.ndarray, joint_state: np.ndarray, endpoint_load: np.ndarray
    ) -> np.ndarray:
        """
        Compute qdd for a batch of states/inputs/endpoint loads.

        Shapes:
          joint_state  : (B, 2*dof)
          inputs (τ)   : (B, dof)
          endpoint_load: (B, space_dim)  # planar force [fx, fy]
          returns      : (B, dof)
        """
        joint_state = self._as_batch(joint_state, self.state_dim)
        inputs = self._as_batch(inputs, self.input_dim)
        endpoint_load = self._as_batch(endpoint_load, self.space_dim)

        q, qd = np.split(joint_state, 2, axis=1)
        self._set_state(q, qd)

        # Library dynamics (cached)
        D = np.asarray(
            inertiaMatrixCOM_cached(self._robot, symbolic=False), dtype=float
        )  # (2,2)
        C = np.asarray(
            centrifugalCoriolisCOM_cached(self._robot, symbolic=False), dtype=float
        )  # (2,2)
        g = np.asarray(
            gravityCOM_cached(self._robot, self._gravity_vec, symbolic=False),
            dtype=float,
        ).reshape(
            1, -1
        )  # (1,2)

        # Geometric Jacobian: take linear x-y rows (cached)
        J = np.asarray(
            geometricJacobian_cached(self._robot, symbolic=False), dtype=float
        )  # (6,2)
        J_xy = J[0:2, :]  # (2,2)

        # External endpoint force -> joint torques with J^T f
        tau_ext = (J_xy.T @ endpoint_load.T).T  # (B,2)
        tau = inputs + tau_ext

        # qdd = D^{-1} [ tau - C qd - g ]
        rhs = tau - (C @ qd.T).T - g
        # Stable linear solve instead of explicit inverse
        qdd = np.linalg.solve(D, rhs.T).T
        return qdd

    def integrate(
        self, dt: float, state_derivative: np.ndarray, joint_state: np.ndarray
    ) -> np.ndarray:
        """Same semi-implicit Euler as base, preserved for compatibility."""
        return super().integrate(dt, state_derivative, joint_state)

    def joint2cartesian(self, joint_state: np.ndarray) -> np.ndarray:
        """
        Returns end-effector cartesian state [x, y, xd, yd] for each batch row.
        """
        joint_state = self._as_batch(joint_state, self.state_dim)
        q, qd = np.split(joint_state, 2, axis=1)
        self._set_state(q, qd)

        # FK frames (cached): we stored (link1, link2, ee)
        link1, link2, T_ee = forwardHTM_cached(self._robot, symbolic=False)
        x, y = float(T_ee[0, 3]), float(T_ee[1, 3])

        # Linear Jacobian (cached)
        J = np.asarray(
            geometricJacobian_cached(self._robot, symbolic=False), dtype=float
        )
        J_xy = J[0:2, :]  # (2,2)
        v_xy = (J_xy @ qd.T).T  # (B,2)

        pos = np.tile(
            [[x, y]], (q.shape[0], 1)
        )  # same pose for a given q across batch row
        return np.concatenate([pos, v_xy], axis=1)

    def path2cartesian(
        self,
        path_coordinates: np.ndarray,  # (B, 2, n_points) or (2, n_points)
        path_fixation_body: np.ndarray,  # (1, n_points) in {0: world, 1: link1, 2: link2}
        joint_state: np.ndarray,  # (B, 2*dof)
    ):
        """
        Transform local fixation points into world XY, along with velocities and d(x,y)/dq.

        Position:
        - Transform local points by world/link1/link2 frames from FK.
        Velocity (approx.):
        - Use the end-effector linear Jacobian for all points (uniform approx).
        Derivative wrt q:
        - Use linear block of the Jacobian as an approximation per point.
        """
        joint_state = self._as_batch(joint_state, self.state_dim)
        q, qd = np.split(joint_state, 2, axis=1)
        batch = q.shape[0]

        # normalize path inputs
        body = np.asarray(path_fixation_body, dtype=int).reshape(1, -1)  # (1, n_points)
        n_points = body.size
        pc = np.asarray(path_coordinates, dtype=float)
        if pc.ndim == 2:
            pc = pc.reshape(1, 2, -1)  # (1, 2, n_points)
        if pc.shape[0] != batch:
            pc = np.repeat(pc, batch, axis=0)  # (batch, 2, n_points)

        # robot state
        self._set_state(q, qd)

        # FK frames: cache returns (link1, link2, ee)
        link1, link2, ee = forwardHTM_cached(self._robot, symbolic=False)
        T_world = np.eye(4)
        T_link1 = link1
        T_link2 = link2

        # select transform per point id -> Ts: (n_points, 4, 4)
        Ts = []
        for b in body.flatten():
            Ts.append(T_world if b == 0 else (T_link1 if b == 1 else T_link2))
        Ts = np.stack(Ts, axis=0)

        # Make homogeneous local points: (batch, 4, n_points)
        ones = np.ones((batch, 1, n_points), dtype=float)
        zeros = np.zeros((batch, 1, n_points), dtype=float)
        local_h = np.concatenate([pc, zeros, ones], axis=1)

        # Apply transforms per point and per batch
        world = np.empty_like(local_h)  # (batch, 4, n_points)
        for i in range(n_points):
            Ti = Ts[i]  # (4,4)
            for b in range(batch):
                world[b, :, i] = Ti @ local_h[b, :, i]  # (4,)

        xy = world[:, 0:2, :]  # (batch, 2, n_points)

        # Approx velocities using EE linear Jacobian (cached)
        J = np.asarray(
            geometricJacobian_cached(self._robot, symbolic=False), dtype=float
        )
        J_lin = J[0:2, :]  # (2,2)
        v_xy = (J_lin @ qd.T).T  # (batch, 2)
        dxy_dt = np.repeat(v_xy[:, :, None], n_points, axis=2)  # (batch, 2, n_points)

        # d(x,y)/dq approximation per point
        dxy_dq = np.repeat(J_lin[None, :, :, None], batch, axis=0)  # (batch, 2, 2, 1)
        dxy_dq = np.repeat(dxy_dq, n_points, axis=3)  # (batch, 2, 2, n_points)

        return xy, dxy_dt, dxy_dq


# ---------------------------------------------------------------------
# Minimal demo (safe to delete if you embed this module)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    arm = TwoDofArm()

    # Optional: configure bounds & dt
    arm.build(
        timestep=0.002,
        pos_upper_bound=[np.deg2rad(140), np.deg2rad(160)],
        pos_lower_bound=[0.0, 0.0],
        vel_upper_bound=[+10.0, +10.0],
        vel_lower_bound=[-10.0, -10.0],
    )

    # State [q1, q2, qd1, qd2] — radians and rad/s
    state = np.array([[np.deg2rad(30.0), np.deg2rad(45.0), 0.1, -0.2]])

    # Inputs (joint torques) and endpoint force [fx, fy]
    u = np.array([[0.0, 0.0]])
    load = np.array([[0.0, 0.0]])

    # qdd from dynamics
    qdd = arm.ode(u, state, load)

    # integrate one step
    new_state = arm.integrate(arm.dt, qdd, state)

    # EE cartesian [x, y, xd, yd]
    cart = arm.joint2cartesian(state)

    # Example path points (two points: one on link1, one on link2)
    path_coords = np.array(
        [[[0.05, 0.02], [0.00, 0.00]]]  # x local
    )  # y local   -> shape (1, 2, 2)
    path_bodies = np.array([[1, 2]])  # 1: link1, 2: link2
    xy, dxy_dt, dxy_dq = arm.path2cartesian(path_coords, path_bodies, state)

    print("qdd:", qdd)
    print("new_state:", new_state)
    print("cartesian [x,y,xd,yd]:", cart)
    print(
        "path xy shape:",
        xy.shape,
        "dxy_dt shape:",
        dxy_dt.shape,
        "dxy_dq shape:",
        dxy_dq.shape,
    )
    print("path xy:", xy)
    print("path dxy_dt:", dxy_dt)
    print("path dxy_dq:", dxy_dq)
