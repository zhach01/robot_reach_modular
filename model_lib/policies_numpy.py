# policies_numpy.py
# Torch-free GRU policies implemented with NumPy only.

from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple


# ---------- helpers ----------


def xavier_uniform(shape, gain=1.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    fan_in, fan_out = shape[1], shape[0]
    a = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(low=-a, high=a, size=shape).astype(np.float32)


def orthogonal(shape, gain=1.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    # For rectangular matrices, do SVD on a random Gaussian
    a = rng.normal(0.0, 1.0, size=shape)
    u, _, vT = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == shape else vT
    return (gain * q[: shape[0], : shape[1]]).astype(np.float32)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x, dtype=np.float64))


def tanh(x):
    return np.tanh(x)


def rect_tanh(x):
    t = np.tanh(x)
    return np.maximum(0.0, t)


def ensure_2d(x: np.ndarray, dim1: int):
    """Ensure (batch, dim1)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    assert x.shape[1] == dim1, f"Expected second dim {dim1}, got {x.shape}"
    return x


def tile_rows(x: np.ndarray, batch: int):
    return np.repeat(x[np.newaxis, :], repeats=batch, axis=0)


# ============================================================
#                    PolicyGRU (NumPy)
# ============================================================


class PolicyGRU_NP:
    """
    NumPy rewrite of your PolicyGRU:
      - GRU with a single time step interface: forward(x, h) -> (u, h_new)
      - x: (batch, input_dim), h: (batch, hidden_dim)
      - u: (batch, output_dim) with sigmoid nonlinearity
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        seed: Optional[int] = None,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = 1
        self.rng = np.random.default_rng(seed)

        # We implement GRU by explicit concatenation [x, h]
        ih = input_dim + hidden_dim

        # Weights follow your original init spirit
        self.Wz = np.concatenate(
            [
                xavier_uniform((hidden_dim, input_dim), rng=self.rng),
                orthogonal((hidden_dim, hidden_dim), rng=self.rng),
            ],
            axis=1,
        )
        self.bz = np.zeros((hidden_dim,), dtype=np.float32)

        self.Wr = np.concatenate(
            [
                xavier_uniform((hidden_dim, input_dim), rng=self.rng),
                orthogonal((hidden_dim, hidden_dim), rng=self.rng),
            ],
            axis=1,
        )
        self.br = np.zeros((hidden_dim,), dtype=np.float32)

        self.Wh = np.concatenate(
            [
                xavier_uniform((hidden_dim, input_dim), rng=self.rng),
                orthogonal((hidden_dim, hidden_dim), rng=self.rng),
            ],
            axis=1,
        )
        self.bh = np.zeros((hidden_dim,), dtype=np.float32)

        self.Wy = xavier_uniform((output_dim, hidden_dim), rng=self.rng)
        self.by = np.full((output_dim,), -3.0, dtype=np.float32)

    def init_hidden(self, batch_size: int) -> np.ndarray:
        return np.zeros((batch_size, self.hidden_dim), dtype=np.float32)

    def forward(self, x: np.ndarray, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        x: (batch, input_dim)
        h: (batch, hidden_dim)
        returns: (u, h_new)
          u: (batch, output_dim), sigmoid output
        """
        x = ensure_2d(x, self.input_dim)
        h = ensure_2d(h, self.hidden_dim)
        batch = x.shape[0]

        concat = np.concatenate([x, h], axis=1)  # (B, I+H)

        z = sigmoid(concat @ self.Wz.T + self.bz)  # (B,H)
        r = sigmoid(concat @ self.Wr.T + self.br)  # (B,H)
        concat_hidden = np.concatenate([x, r * h], axis=1)  # (B, I+H)
        h_tilde = tanh(concat_hidden @ self.Wh.T + self.bh)  # (B,H)
        h_new = (1.0 - z) * h + z * h_tilde  # (B,H)

        u = sigmoid(h_new @ self.Wy.T + self.by)  # (B,O)
        return u.astype(np.float32), h_new.astype(np.float32)


# ============================================================
#                ModularPolicyGRU (NumPy)
# ============================================================


class ModularPolicyGRU_NP:
    """
    NumPy rewrite of your ModularPolicyGRU.

    Major features preserved:
      - Modules with sizes in `module_size`
      - Masks for inputs (vision/proprio/task), output masks per module
      - Random sparse inter-module recurrent connectivity (connectivity_mask prob)
      - Optional integer delays per (post, pre) module: connectivity_delay[i,j]
      - Optional output delay (use hidden buffer)
      - Optional E/I (Dale) constraints via proportion_excitatory per module
      - Optional spectral scaling of recurrent block
      - Optional cancelation_matrix and time-based cancellation

    Forward() is single-step:
      y, h_new = forward(x, h_prev)
        x: (batch, input_size)
        h_prev: (batch, hidden_size)

    Initialize with `init_hidden(batch)` to set hidden buffer (for delays).
    """

    def __init__(
        self,
        input_size: int,
        module_size: List[int],
        output_size: int,
        vision_mask: List[float],
        proprio_mask: List[float],
        task_mask: List[float],
        connectivity_mask: np.ndarray,
        output_mask: List[float],
        vision_dim: List[int],
        proprio_dim: List[int],
        task_dim: List[int],
        connectivity_delay: np.ndarray,
        spectral_scaling: Optional[float] = None,
        proportion_excitatory: Optional[List[float]] = None,
        input_gain: float = 1.0,
        seed: Optional[int] = None,
        activation: str = "tanh",
        output_delay: int = 0,
        cancelation_matrix: Optional[np.ndarray] = None,
        last_task_proprio_only: bool = False,
    ):
        # store
        self.rng = np.random.default_rng(seed)
        self.input_size = input_size
        self.module_size = module_size
        self.hidden_size = int(np.sum(module_size))
        self.output_size = output_size
        self.num_modules = len(module_size)

        # activation
        assert activation in ("tanh", "rect_tanh")
        self.activation_name = activation
        self._act = tanh if activation == "tanh" else rect_tanh

        # delays
        self.connectivity_delay = np.array(connectivity_delay, dtype=np.int32)
        self.output_delay = int(output_delay)
        self.max_connectivity_delay = (
            int(np.max(self.connectivity_delay)) if self.connectivity_delay.size else 0
        )
        self.max_delay = int(max(self.max_connectivity_delay, self.output_delay))
        self.h_buffer = None  # will be (batch, hidden, max_delay+1)
        self.counter = 0
        self.cancel_times = None

        # optional cancellation
        self.cancelation_matrix = (
            None
            if cancelation_matrix is None
            else np.asarray(cancelation_matrix, dtype=np.float32)
        )

        # dims for modules
        self.module_dims: List[np.ndarray] = []
        cur = 0
        for sz in module_size:
            self.module_dims.append(np.arange(cur, cur + sz))
            cur += sz

        # sanity checks
        assert len(vision_mask) == self.num_modules
        assert len(proprio_mask) == self.num_modules
        assert len(task_mask) == self.num_modules
        assert connectivity_mask.shape == (self.num_modules, self.num_modules)
        assert len(output_mask) == self.num_modules
        assert len(vision_dim) + len(proprio_dim) + len(task_dim) == self.input_size
        if proportion_excitatory is not None:
            assert len(proportion_excitatory) == self.num_modules

        # --------- Build sparse probability mask for GRU big weights ---------
        # Wz, Wr, Wh are (H, I+H). We'll construct a probability mask then sample binary mask from it.
        prob = np.zeros(
            (self.hidden_size, self.input_size + self.hidden_size), dtype=np.float32
        )

        for i_mod in range(self.num_modules):
            rows = self.module_dims[i_mod]

            # inputs
            if len(vision_dim) > 0:
                prob[np.ix_(rows, vision_dim)] = float(vision_mask[i_mod])
            if len(proprio_dim) > 0:
                prob[np.ix_(rows, proprio_dim)] = float(proprio_mask[i_mod])

            if len(task_dim) > 0:
                if last_task_proprio_only:
                    if len(task_dim) > 1:
                        general_task_dims = task_dim[:-1]
                        prob[np.ix_(rows, general_task_dims)] = float(task_mask[i_mod])
                    last_task_idx = task_dim[-1]
                    prob[np.ix_(rows, [last_task_idx])] = float(proprio_mask[i_mod])
                else:
                    prob[np.ix_(rows, task_dim)] = float(task_mask[i_mod])

            # recurrent (select a subset from module j columns)
            for j_mod in range(self.num_modules):
                p = float(connectivity_mask[i_mod, j_mod])
                if p <= 0:
                    continue
                pre_cols = self.module_dims[j_mod] + self.input_size  # offset for [x,h]
                n_pre = len(pre_cols)
                n_take = int(np.ceil(p * n_pre))
                chosen = self.rng.choice(pre_cols, size=n_take, replace=False)
                prob[np.ix_(rows, chosen)] = 1.0  # 1.0 prob for chosen subset

        # sample binary connectivity masks
        mask_connectivity = self.rng.binomial(1, prob).astype(np.float32)

        # output mask: (O,H)
        y_prob = np.zeros((self.output_size, self.hidden_size), dtype=np.float32)
        for j_mod in range(self.num_modules):
            cols = self.module_dims[j_mod]
            y_prob[:, cols] = float(output_mask[j_mod])
        mask_Y = self.rng.binomial(1, y_prob).astype(np.float32)

        # --------- Initialize parameters ---------
        ih = self.input_size + self.hidden_size
        gain = input_gain

        self.h0 = np.zeros((1, self.hidden_size), dtype=np.float32)

        self.Wz = (
            np.concatenate(
                [
                    xavier_uniform(
                        (self.hidden_size, self.input_size), gain=gain, rng=self.rng
                    ),
                    self.rng.normal(
                        0,
                        1 / np.sqrt(self.hidden_size),
                        size=(self.hidden_size, self.hidden_size),
                    ).astype(np.float32),
                ],
                axis=1,
            )
            * mask_connectivity
        )
        self.bz = np.zeros((self.hidden_size,), dtype=np.float32)

        self.Wr = (
            np.concatenate(
                [
                    xavier_uniform(
                        (self.hidden_size, self.input_size), gain=gain, rng=self.rng
                    ),
                    self.rng.normal(
                        0,
                        1 / np.sqrt(self.hidden_size),
                        size=(self.hidden_size, self.hidden_size),
                    ).astype(np.float32),
                ],
                axis=1,
            )
            * mask_connectivity
        )
        self.br = np.zeros((self.hidden_size,), dtype=np.float32)

        self.Wh = (
            np.concatenate(
                [
                    xavier_uniform(
                        (self.hidden_size, self.input_size), gain=gain, rng=self.rng
                    ),
                    self.rng.normal(
                        0,
                        1 / np.sqrt(self.hidden_size),
                        size=(self.hidden_size, self.hidden_size),
                    ).astype(np.float32),
                ],
                axis=1,
            )
            * mask_connectivity
        )
        self.bh = np.zeros((self.hidden_size,), dtype=np.float32)

        self.Y = (
            xavier_uniform((self.output_size, self.hidden_size), rng=self.rng) * mask_Y
        )
        self.bY = np.full((self.output_size,), -3.0, dtype=np.float32)

        self.mask_Wz = mask_connectivity
        self.mask_Wr = mask_connectivity
        self.mask_Wh = mask_connectivity
        self.mask_Y = mask_Y

        self.Wz_cached = self.Wz.copy()
        self.Wr_cached = self.Wr.copy()
        self.Wh_cached = self.Wh.copy()
        self.Y_cached = self.Y.copy()

        # --------- Optional Dale's law enforcement ---------
        self.unittype_W = None
        if proportion_excitatory is not None:
            # per-module E/I assignment
            self.unittype_W = np.zeros(
                (self.hidden_size, self.hidden_size), dtype=np.float32
            )
            for m, p_exc in enumerate(proportion_excitatory):
                idx = self.module_dims[m]
                # excit=+1, inhib=-1
                types = (self.rng.binomial(1, p_exc, size=len(idx)) * 2 - 1).astype(
                    np.float32
                )
                self.unittype_W[:, idx] = types[np.newaxis, :]

            self.enforce_dale(zero_out=False)

        # --------- Optional spectral scaling ---------
        if spectral_scaling is not None and spectral_scaling > 0:
            # scale only the recurrent HxH part of Wh
            Wh_in, Wh_rec = self.Wh[:, : self.input_size], self.Wh[:, self.input_size :]
            if np.any(Wh_rec != 0):
                try:
                    eigvals = np.linalg.eigvals(Wh_rec)
                    eig_norm = np.max(np.abs(eigvals))
                    if eig_norm > 1e-6:
                        Wh_rec = (spectral_scaling * (Wh_rec / eig_norm)).astype(
                            np.float32
                        )
                        self.Wh = np.concatenate([Wh_in, Wh_rec], axis=1)
                except np.linalg.LinAlgError:
                    pass  # keep as-is if decomposition fails

    # ---------- public API ----------

    def set_cancelation_matrix(self, cancelation_matrix: np.ndarray):
        self.cancelation_matrix = np.asarray(cancelation_matrix, dtype=np.float32)

    def reset_counter(self):
        self.counter = 0

    def set_cancel_times(self, times: List[int]):
        self.cancel_times = set(int(t) for t in times)

    def init_hidden(self, batch_size: int) -> np.ndarray:
        h0 = np.tile(self._activation(self.h0), (batch_size, 1))
        # hidden buffer for delays: newest at index 0
        self.h_buffer = np.tile(h0[:, :, None], (1, 1, self.max_delay + 1))
        return h0

    def forward(
        self, x: np.ndarray, h_prev: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        GRU forward with optional inter-module delays and output delay.
        - x: (B, I)
        - h_prev: (B, H)
        returns:
        y: (B, O)
        h_new: (B, H)
        """

        B = x.shape[0]

        # --- update hidden buffer: newest previous state at index 0
        # h_buffer shape: (B, H, max_delay+1)
        if self.h_buffer is None or len(self.h_buffer) == 0:
            # initialize on first call
            self.h_buffer = np.tile(h_prev[:, :, None], (1, 1, int(self.max_delay) + 1))
        else:
            self.h_buffer = np.concatenate(
                [h_prev[:, :, None], self.h_buffer[:, :, :-1]], axis=2
            )

        if self.max_connectivity_delay > 0:
            # -------- delayed path: compute per post-module i with correct (i,j) delays --------
            H = self.hidden_size
            h_new = np.zeros_like(h_prev, dtype=np.float32)

            for i in range(self.num_modules):
                rows = self.module_dims[i]  # indices in output (post) module i
                # build delayed hidden state specific to post-module i
                h_prev_delayed_i = np.zeros_like(h_prev, dtype=np.float32)
                for j in range(self.num_modules):
                    cols_j = self.module_dims[j]  # presynaptic (pre) module j indices
                    d_ij = int(self.connectivity_delay[i, j])  # delay from j->i
                    h_prev_delayed_i[:, cols_j] = self.h_buffer[:, cols_j, d_ij]

                # full z, r for all hidden units using the i-specific delayed hidden
                # W* shapes: (H, I+H) -> split once
                Wz_in, Wz_hh = (
                    self.Wz[:, : self.input_size],
                    self.Wz[:, self.input_size :],
                )
                Wr_in, Wr_hh = (
                    self.Wr[:, : self.input_size],
                    self.Wr[:, self.input_size :],
                )
                Wh_in, Wh_hh = (
                    self.Wh[:, : self.input_size],
                    self.Wh[:, self.input_size :],
                )

                preact_z_full = (
                    x @ Wz_in.T + h_prev_delayed_i @ Wz_hh.T + self.bz
                )  # (B, H)
                preact_r_full = (
                    x @ Wr_in.T + h_prev_delayed_i @ Wr_hh.T + self.br
                )  # (B, H)
                z_full = 1.0 / (1.0 + np.exp(-preact_z_full))  # (B, H)
                r_full = 1.0 / (1.0 + np.exp(-preact_r_full))  # (B, H)

                # candidate preactivation for the selected rows:
                #   x @ Wh_in[rows].T  -> (B, |rows|)
                # + (r_full ∘ h_prev_delayed_i) @ Wh_hh[rows].T -> (B, |rows|)
                # + bias rows
                h_tilde_in_rows = (
                    x @ Wh_in[rows, :].T
                    + (r_full * h_prev_delayed_i) @ Wh_hh[rows, :].T
                    + self.bh[rows]
                )  # (B, |rows|)
                h_tilde_rows = self._activation(h_tilde_in_rows)

                # gate rows for z and update with delayed hidden (per GRU eqs.)
                z_rows = z_full[:, rows]  # (B, |rows|)
                h_new[:, rows] = (1.0 - z_rows) * h_prev_delayed_i[
                    :, rows
                ] + z_rows * h_tilde_rows

        else:
            # -------- no-delay fast path: single full GRU pass --------
            Wz_in, Wz_hh = self.Wz[:, : self.input_size], self.Wz[:, self.input_size :]
            Wr_in, Wr_hh = self.Wr[:, : self.input_size], self.Wr[:, self.input_size :]
            Wh_in, Wh_hh = self.Wh[:, : self.input_size], self.Wh[:, self.input_size :]

            preact_z = x @ Wz_in.T + h_prev @ Wz_hh.T + self.bz  # (B, H)
            preact_r = x @ Wr_in.T + h_prev @ Wr_hh.T + self.br  # (B, H)
            z = 1.0 / (1.0 + np.exp(-preact_z))
            r = 1.0 / (1.0 + np.exp(-preact_r))

            h_tilde_in = x @ Wh_in.T + (r * h_prev) @ Wh_hh.T + self.bh
            h_tilde = self._activation(h_tilde_in)

            h_new = (1.0 - z) * h_prev + z * h_tilde

        # optional cancellation “pulse”
        if (
            getattr(self, "cancelation_matrix", None) is not None
            and getattr(self, "cancel_times", None) is not None
        ):
            if self.counter in self.cancel_times:
                # h_new += h_new @ C
                h_new = h_new + h_new @ self.cancelation_matrix

        # output (with optional delay)
        if self.output_delay == 0:
            y = 1.0 / (1.0 + np.exp(-(h_new @ self.Y.T + self.bY)))
        else:
            # newest previous state is buffer[:,:,0]; use kth-old => index (output_delay-1)
            k = int(self.output_delay) - 1
            h_for_out = self.h_buffer[:, :, k]
            y = 1.0 / (1.0 + np.exp(-(h_for_out @ self.Y.T + self.bY)))

        self.counter += 1
        return y, h_new

    # ---------- tools ----------

    def cache_policy(self):
        self.Wr_cached = self.Wr.copy()
        self.Wz_cached = self.Wz.copy()
        self.Wh_cached = self.Wh.copy()
        self.Y_cached = self.Y.copy()

    def enforce_dale(self, zero_out: bool = False):
        """
        Enforce Dale on the HxH block of (Wr, Wz, Wh) according to self.unittype_W (+1 exc, -1 inh).
        Keeps input->hidden part unchanged.
        """
        if self.unittype_W is None:
            return

        def _apply(mat, mat_cached):
            W_in, W_rec = mat[:, : self.input_size], mat[:, self.input_size :]
            Wc_in, Wc_rec = (
                mat_cached[:, : self.input_size],
                mat_cached[:, self.input_size :],
            )
            # enforce sign per unittype columns
            # unittype dims match (H,H) for recurrent columns
            exc_mask = self.unittype_W == 1.0
            inh_mask = self.unittype_W == -1.0
            if zero_out:
                W_rec[(W_rec < 0) & exc_mask] = 0.0
                W_rec[(W_rec > 0) & inh_mask] = 0.0
            else:
                W_rec[(W_rec < 0) & exc_mask] = np.abs(Wc_rec[(W_rec < 0) & exc_mask])
                W_rec[(W_rec > 0) & inh_mask] = -np.abs(Wc_rec[(W_rec > 0) & inh_mask])
            return np.concatenate([W_in, W_rec], axis=1)

        self.Wr = _apply(self.Wr, self.Wr_cached)
        self.Wz = _apply(self.Wz, self.Wz_cached)
        self.Wh = _apply(self.Wh, self.Wh_cached)
        self.cache_policy()

    # internal
    def _activation(self, x):
        return self._act(x)


# ============================================================
#                      Smoke test
# ============================================================
if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    # ---- Simple PolicyGRU_NP ----
    print("\n=== PolicyGRU_NP ===")
    B, I, H, O = 2, 5, 16, 3
    pol = PolicyGRU_NP(I, H, O, seed=0)
    h = pol.init_hidden(B)
    x = np.random.default_rng(1).normal(size=(B, I)).astype(np.float32)
    u, h = pol.forward(x, h)
    print("u shape:", u.shape, "| h shape:", h.shape, "| u[0]:", u[0])

    # ---- ModularPolicyGRU_NP ----
    print("\n=== ModularPolicyGRU_NP ===")
    # toy config: 3 modules
    module_size = [8, 8, 8]
    input_size = 6
    output_size = 4
    # split inputs: vision(2), proprio(2), task(2)
    vision_dim = [0, 1]
    proprio_dim = [2, 3]
    task_dim = [4, 5]

    vision_mask = [1.0, 0.5, 0.0]  # probabilities for input->module
    proprio_mask = [0.0, 0.5, 1.0]
    task_mask = [0.5, 0.5, 0.5]

    # inter-module connectivity probabilities (post x pre)
    connectivity_mask = np.array(
        [
            [0.2, 0.1, 0.0],
            [0.1, 0.2, 0.1],
            [0.0, 0.1, 0.2],
        ],
        dtype=np.float32,
    )

    # output mask per module (probability that module contributes to outputs)
    output_mask = [1.0, 0.5, 0.2]

    # delays (post x pre)
    connectivity_delay = np.array(
        [
            [0, 1, 0],
            [0, 0, 2],
            [0, 0, 0],
        ],
        dtype=int,
    )

    net = ModularPolicyGRU_NP(
        input_size=input_size,
        module_size=module_size,
        output_size=output_size,
        vision_mask=vision_mask,
        proprio_mask=proprio_mask,
        task_mask=task_mask,
        connectivity_mask=connectivity_mask,
        output_mask=output_mask,
        vision_dim=vision_dim,
        proprio_dim=proprio_dim,
        task_dim=task_dim,
        connectivity_delay=connectivity_delay,
        spectral_scaling=0.99,
        proportion_excitatory=[0.8, 0.8, 0.8],
        input_gain=1.0,
        seed=42,
        activation="tanh",
        output_delay=1,
        cancelation_matrix=None,
        last_task_proprio_only=False,
    )

    B = 2
    h = net.init_hidden(B)
    x = np.random.default_rng(2).normal(size=(B, input_size)).astype(np.float32)

    for t in range(5):
        y, h = net.forward(x, h)
        print(f"t={t}  y[0]:", y[0])

    print("\nSmoke tests complete ✓")
