# policies_torch.py
# -*- coding: utf-8 -*-
"""
PyTorch GRU policies (Torch counterpart of policies_numpy.py).

- PolicyGRU_Torch: simple GRU cell with one-step interface.
- ModularPolicyGRU_Torch: modular GRU with sparse connectivity,
  delays, optional Dale's law, spectral scaling, cancellation pulses.

All math is pure Torch:
- Batchable on leading dimension.
- GPU-safe via device + dtype handling.
- Fully differentiable through forward paths.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
from torch import Tensor, nn


# ---------- helpers ----------


def xavier_uniform_torch(
    shape: Tuple[int, int],
    gain: float = 1.0,
    generator: Optional[torch.Generator] = None,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Torch equivalent of NumPy xavier_uniform."""
    fan_out, fan_in = shape[0], shape[1]
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    t = torch.empty(shape, device=device, dtype=dtype)
    if generator is not None:
        t.uniform_(-a, a, generator=generator)
    else:
        t.uniform_(-a, a)
    return t


def orthogonal_torch(
    shape: Tuple[int, int],
    gain: float = 1.0,
    generator: Optional[torch.Generator] = None,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Torch equivalent of NumPy orthogonal init."""
    # work in float64 for SVD stability, cast back at the end
    a = torch.empty(shape, device=device, dtype=torch.float64)
    if generator is not None:
        a.normal_(mean=0.0, std=1.0, generator=generator)
    else:
        a.normal_(mean=0.0, std=1.0)
    # full_matrices=False so shapes match min(m,n)
    u, _, vT = torch.linalg.svd(a, full_matrices=False)
    q = u if u.shape == a.shape else vT
    q = q[: shape[0], : shape[1]]
    q = gain * q
    return q.to(dtype=dtype)


def rect_tanh_torch(x: Tensor) -> Tensor:
    """Rectified tanh: max(0, tanh(x))."""
    return torch.relu(torch.tanh(x))


def ensure_2d_torch(
    x: Union[Tensor, float, List[float]],
    dim1: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Ensure x is (batch, dim1) Torch tensor on given device/dtype."""
    t = torch.as_tensor(x, dtype=dtype, device=device)
    if t.dim() == 1:
        t = t.view(1, -1)
    if t.dim() != 2 or t.shape[1] != dim1:
        raise ValueError(f"Expected shape (*,{dim1}), got {tuple(t.shape)}")
    return t


# ============================================================
#                    PolicyGRU (Torch)
# ============================================================


class PolicyGRU_Torch(nn.Module):
    """
    Torch rewrite of PolicyGRU_NP:
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
        device: Union[str, torch.device] = "cpu",
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = 1

        self.device = torch.device(device)
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()

        # local RNG for initialization
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(int(seed))

        ih = input_dim + hidden_dim  # concat [x,h]

        # Wz, Wr, Wh: (H, I+H)
        Wz_in = xavier_uniform_torch(
            (hidden_dim, input_dim),
            generator=self.generator,
            device=self.device,
            dtype=self.dtype,
        )
        Wz_rec = orthogonal_torch(
            (hidden_dim, hidden_dim),
            generator=self.generator,
            device=self.device,
            dtype=self.dtype,
        )
        Wz_full = torch.cat([Wz_in, Wz_rec], dim=1)

        Wr_in = xavier_uniform_torch(
            (hidden_dim, input_dim),
            generator=self.generator,
            device=self.device,
            dtype=self.dtype,
        )
        Wr_rec = orthogonal_torch(
            (hidden_dim, hidden_dim),
            generator=self.generator,
            device=self.device,
            dtype=self.dtype,
        )
        Wr_full = torch.cat([Wr_in, Wr_rec], dim=1)

        Wh_in = xavier_uniform_torch(
            (hidden_dim, input_dim),
            generator=self.generator,
            device=self.device,
            dtype=self.dtype,
        )
        Wh_rec = orthogonal_torch(
            (hidden_dim, hidden_dim),
            generator=self.generator,
            device=self.device,
            dtype=self.dtype,
        )
        Wh_full = torch.cat([Wh_in, Wh_rec], dim=1)

        self.Wz = nn.Parameter(Wz_full)  # (H, I+H)
        self.bz = nn.Parameter(torch.zeros(hidden_dim, device=self.device, dtype=self.dtype))

        self.Wr = nn.Parameter(Wr_full)
        self.br = nn.Parameter(torch.zeros(hidden_dim, device=self.device, dtype=self.dtype))

        self.Wh = nn.Parameter(Wh_full)
        self.bh = nn.Parameter(torch.zeros(hidden_dim, device=self.device, dtype=self.dtype))

        self.Wy = nn.Parameter(
            xavier_uniform_torch(
                (output_dim, hidden_dim),
                generator=self.generator,
                device=self.device,
                dtype=self.dtype,
            )
        )
        self.by = nn.Parameter(
            torch.full((output_dim,), -3.0, device=self.device, dtype=self.dtype)
        )

    def init_hidden(self, batch_size: int) -> Tensor:
        """Return initial hidden state (zeros)."""
        return torch.zeros(
            (batch_size, self.hidden_dim), device=self.device, dtype=self.dtype
        )

    def forward(self, x: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        """
        x: (batch, input_dim)
        h: (batch, hidden_dim)
        returns: (u, h_new)
          u: (batch, output_dim), sigmoid output
        """
        x = ensure_2d_torch(x, self.input_dim, device=self.device, dtype=self.dtype)
        h = ensure_2d_torch(h, self.hidden_dim, device=self.device, dtype=self.dtype)

        concat = torch.cat([x, h], dim=1)  # (B, I+H)

        z = torch.sigmoid(concat @ self.Wz.t() + self.bz)  # (B,H)
        r = torch.sigmoid(concat @ self.Wr.t() + self.br)  # (B,H)
        concat_hidden = torch.cat([x, r * h], dim=1)       # (B, I+H)
        h_tilde = torch.tanh(concat_hidden @ self.Wh.t() + self.bh)  # (B,H)
        h_new = (1.0 - z) * h + z * h_tilde               # (B,H)

        u = torch.sigmoid(h_new @ self.Wy.t() + self.by)  # (B,O)
        return u, h_new


# ============================================================
#                ModularPolicyGRU (Torch)
# ============================================================


class ModularPolicyGRU_Torch(nn.Module):
    """
    Torch rewrite of ModularPolicyGRU_NP.

    Preserves:
      - Modules with sizes in `module_size`
      - Masks for inputs (vision/proprio/task), output masks per module
      - Random sparse inter-module recurrent connectivity (connectivity_mask prob)
      - Optional integer delays per (post, pre) module: connectivity_delay[i,j]
      - Optional output delay (use hidden buffer)
      - Optional E/I (Dale) constraints via proportion_excitatory per module
      - Optional spectral scaling of recurrent block
      - Optional cancelation_matrix and time-based cancellation

    Forward is single-step:
      y, h_new = forward(x, h_prev)
        x: (batch, input_size)
        h_prev: (batch, hidden_size)
    """

    def __init__(
        self,
        input_size: int,
        module_size: List[int],
        output_size: int,
        vision_mask: List[float],
        proprio_mask: List[float],
        task_mask: List[float],
        connectivity_mask: Union[Tensor, Any],
        output_mask: List[float],
        vision_dim: List[int],
        proprio_dim: List[int],
        task_dim: List[int],
        connectivity_delay: Union[Tensor, Any],
        spectral_scaling: Optional[float] = None,
        proportion_excitatory: Optional[List[float]] = None,
        input_gain: float = 1.0,
        seed: Optional[int] = None,
        activation: str = "tanh",
        output_delay: int = 0,
        cancelation_matrix: Optional[Tensor] = None,
        last_task_proprio_only: bool = False,
        device: Union[str, torch.device] = "cpu",
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        # basic dims
        self.device = torch.device(device)
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.input_size = input_size
        self.module_size = module_size
        self.hidden_size = int(sum(module_size))
        self.output_size = output_size
        self.num_modules = len(module_size)

        # RNG
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(int(seed))

        # activation
        assert activation in ("tanh", "rect_tanh")
        self.activation_name = activation
        self._act = torch.tanh if activation == "tanh" else rect_tanh_torch

        # delays
        self.connectivity_delay = torch.as_tensor(
            connectivity_delay, dtype=torch.int64, device=self.device
        )
        self.output_delay = int(output_delay)
        self.max_connectivity_delay = (
            int(self.connectivity_delay.max().item())
            if self.connectivity_delay.numel() > 0
            else 0
        )
        self.max_delay = int(max(self.max_connectivity_delay, self.output_delay))
        self.register_buffer(
            "h_buffer", None, persistent=False
        )  # will become (B,H,max_delay+1)
        self.counter: int = 0
        self.cancel_times: Optional[set[int]] = None

        # optional cancellation matrix (H x H)
        if cancelation_matrix is not None:
            cm = torch.as_tensor(
                cancelation_matrix, dtype=self.dtype, device=self.device
            )
        else:
            cm = None
        self.register_buffer("cancelation_matrix", cm, persistent=False)

        # module dims as index tensors
        self.module_dims: List[Tensor] = []
        cur = 0
        for sz in module_size:
            idx = torch.arange(cur, cur + sz, device=self.device, dtype=torch.long)
            self.module_dims.append(idx)
            cur += sz

        # sanity checks
        assert len(vision_mask) == self.num_modules
        assert len(proprio_mask) == self.num_modules
        assert len(task_mask) == self.num_modules
        connectivity_mask_t = torch.as_tensor(
            connectivity_mask, dtype=self.dtype, device=self.device
        )
        assert connectivity_mask_t.shape == (self.num_modules, self.num_modules)
        assert len(output_mask) == self.num_modules
        vision_dim_t = torch.as_tensor(vision_dim, dtype=torch.long, device=self.device)
        proprio_dim_t = torch.as_tensor(
            proprio_dim, dtype=torch.long, device=self.device
        )
        task_dim_t = torch.as_tensor(task_dim, dtype=torch.long, device=self.device)
        assert (
            vision_dim_t.numel()
            + proprio_dim_t.numel()
            + task_dim_t.numel()
            == self.input_size
        )
        if proportion_excitatory is not None:
            assert len(proportion_excitatory) == self.num_modules

        # --------- Build probability mask (H, I+H) ---------
        prob = torch.zeros(
            (self.hidden_size, self.input_size + self.hidden_size),
            device=self.device,
            dtype=self.dtype,
        )

        # fill input parts
        for i_mod in range(self.num_modules):
            rows = self.module_dims[i_mod]
            vmask = float(vision_mask[i_mod])
            pmask = float(proprio_mask[i_mod])
            tmask = float(task_mask[i_mod])

            # vision input dims
            if vision_dim_t.numel() > 0:
                for r in rows.tolist():
                    prob[r, vision_dim_t] = vmask

            # proprio input dims
            if proprio_dim_t.numel() > 0:
                for r in rows.tolist():
                    prob[r, proprio_dim_t] = pmask

            # task dims (with optional last_task_proprio_only)
            if task_dim_t.numel() > 0:
                if last_task_proprio_only and task_dim_t.numel() > 1:
                    general_task_dims = task_dim_t[:-1]
                    last_task_idx = task_dim_t[-1]
                    for r in rows.tolist():
                        prob[r, general_task_dims] = tmask
                        prob[r, last_task_idx] = pmask
                else:
                    for r in rows.tolist():
                        prob[r, task_dim_t] = tmask

        # recurrent connections (post x pre)
        for i_mod in range(self.num_modules):
            rows = self.module_dims[i_mod]
            for j_mod in range(self.num_modules):
                p = float(connectivity_mask_t[i_mod, j_mod].item())
                if p <= 0.0:
                    continue
                pre_cols = self.module_dims[j_mod] + self.input_size  # offset by I
                n_pre = pre_cols.numel()
                n_take = int(math.ceil(p * n_pre))
                if n_take <= 0:
                    continue
                perm = torch.randperm(n_pre, generator=self.generator, device=self.device)
                chosen = pre_cols[perm[:n_take]]
                for r in rows.tolist():
                    prob[r, chosen] = 1.0  # 1.0 probability for these connections

        # sample binary connectivity mask (H, I+H)
        rand_conn = torch.rand(
            prob.shape,
            generator=self.generator,
            device=self.device,
            dtype=self.dtype,
        )
        mask_connectivity = (rand_conn < prob).to(self.dtype)

        # output mask (O, H)
        y_prob = torch.zeros(
            (self.output_size, self.hidden_size),
            device=self.device,
            dtype=self.dtype,
        )
        for j_mod in range(self.num_modules):
            cols = self.module_dims[j_mod]
            p_out = float(output_mask[j_mod])
            for c in cols.tolist():
                y_prob[:, c] = p_out

        rand_y = torch.rand(
            y_prob.shape,
            generator=self.generator,
            device=self.device,
            dtype=self.dtype,
        )
        mask_Y = (rand_y < y_prob).to(self.dtype)

        # register masks as buffers
        self.register_buffer("mask_Wz", mask_connectivity.clone(), persistent=False)
        self.register_buffer("mask_Wr", mask_connectivity.clone(), persistent=False)
        self.register_buffer("mask_Wh", mask_connectivity.clone(), persistent=False)
        self.register_buffer("mask_Y", mask_Y.clone(), persistent=False)

        # --------- Initialize parameters (same structure as NumPy) ---------
        gain = float(input_gain)

        self.register_buffer(
            "h0", torch.zeros((1, self.hidden_size), device=self.device, dtype=self.dtype)
        )

        # Wz
        Wz_in = xavier_uniform_torch(
            (self.hidden_size, self.input_size),
            gain=gain,
            generator=self.generator,
            device=self.device,
            dtype=self.dtype,
        )
        Wz_rec = torch.normal(
            mean=0.0,
            std=1.0 / math.sqrt(self.hidden_size),
            size=(self.hidden_size, self.hidden_size),
            generator=self.generator,
            device=self.device,
            dtype=self.dtype,
        )
        Wz_full = torch.cat([Wz_in, Wz_rec], dim=1) * self.mask_Wz
        self.Wz = nn.Parameter(Wz_full)
        self.bz = nn.Parameter(torch.zeros(self.hidden_size, device=self.device, dtype=self.dtype))

        # Wr
        Wr_in = xavier_uniform_torch(
            (self.hidden_size, self.input_size),
            gain=gain,
            generator=self.generator,
            device=self.device,
            dtype=self.dtype,
        )
        Wr_rec = torch.normal(
            mean=0.0,
            std=1.0 / math.sqrt(self.hidden_size),
            size=(self.hidden_size, self.hidden_size),
            generator=self.generator,
            device=self.device,
            dtype=self.dtype,
        )
        Wr_full = torch.cat([Wr_in, Wr_rec], dim=1) * self.mask_Wr
        self.Wr = nn.Parameter(Wr_full)
        self.br = nn.Parameter(torch.zeros(self.hidden_size, device=self.device, dtype=self.dtype))

        # Wh
        Wh_in = xavier_uniform_torch(
            (self.hidden_size, self.input_size),
            gain=gain,
            generator=self.generator,
            device=self.device,
            dtype=self.dtype,
        )
        Wh_rec = torch.normal(
            mean=0.0,
            std=1.0 / math.sqrt(self.hidden_size),
            size=(self.hidden_size, self.hidden_size),
            generator=self.generator,
            device=self.device,
            dtype=self.dtype,
        )
        Wh_full = torch.cat([Wh_in, Wh_rec], dim=1) * self.mask_Wh
        self.Wh = nn.Parameter(Wh_full)
        self.bh = nn.Parameter(torch.zeros(self.hidden_size, device=self.device, dtype=self.dtype))

        # Y
        Y_full = (
            xavier_uniform_torch(
                (self.output_size, self.hidden_size),
                generator=self.generator,
                device=self.device,
                dtype=self.dtype,
            )
            * self.mask_Y
        )
        self.Y = nn.Parameter(Y_full)
        self.bY = nn.Parameter(
            torch.full(
                (self.output_size,),
                -3.0,
                device=self.device,
                dtype=self.dtype,
            )
        )

        # cached copies for Dale enforcement
        self.register_buffer("Wz_cached", self.Wz.data.clone(), persistent=False)
        self.register_buffer("Wr_cached", self.Wr.data.clone(), persistent=False)
        self.register_buffer("Wh_cached", self.Wh.data.clone(), persistent=False)
        self.register_buffer("Y_cached", self.Y.data.clone(), persistent=False)

        # --------- Optional Dale's law enforcement ---------
        self.register_buffer("unittype_W", None, persistent=False)
        if proportion_excitatory is not None:
            # per-module E/I assignment
            ut = torch.zeros(
                (self.hidden_size, self.hidden_size),
                device=self.device,
                dtype=self.dtype,
            )
            for m, p_exc in enumerate(proportion_excitatory):
                idx = self.module_dims[m]
                # excit=+1, inhib=-1
                probs = torch.full(
                    (idx.numel(),),
                    float(p_exc),
                    device=self.device,
                    dtype=self.dtype,
                )
                bern = torch.bernoulli(probs, generator=self.generator)
                types = bern * 2.0 - 1.0
                ut[:, idx] = types.unsqueeze(0)
            self.unittype_W = ut
            self.enforce_dale(zero_out=False)

        # --------- Optional spectral scaling ---------
        if spectral_scaling is not None and spectral_scaling > 0:
            with torch.no_grad():
                Wh_in = self.Wh[:, : self.input_size]
                Wh_rec = self.Wh[:, self.input_size :]
                if torch.any(Wh_rec != 0):
                    try:
                        eigvals = torch.linalg.eigvals(Wh_rec.to(torch.complex128))
                        eig_norm = eigvals.abs().max().real
                        if eig_norm > 1e-6:
                            Wh_rec_scaled = (
                                float(spectral_scaling) * (Wh_rec / eig_norm)
                            ).to(self.dtype)
                            self.Wh.data = torch.cat([Wh_in, Wh_rec_scaled], dim=1)
                    except RuntimeError:
                        # keep as-is if eigen decomposition fails
                        pass
                self.Wh_cached = self.Wh.data.clone()

    # ---------- public API ----------

    def set_cancelation_matrix(self, cancelation_matrix: Tensor):
        with torch.no_grad():
            cm = torch.as_tensor(
                cancelation_matrix, dtype=self.dtype, device=self.device
            )
            self.cancelation_matrix = cm

    def reset_counter(self):
        self.counter = 0

    def set_cancel_times(self, times: List[int]):
        self.cancel_times = set(int(t) for t in times)

    def init_hidden(self, batch_size: int) -> Tensor:
        """
        Initialize hidden state and buffer.
        Returns h0: (batch, hidden_size)
        """
        h0 = self._activation(self.h0).expand(batch_size, -1).contiguous()
        if self.max_delay > 0:
            self.h_buffer = h0.unsqueeze(-1).expand(
                batch_size, self.hidden_size, self.max_delay + 1
            ).contiguous()
        else:
            self.h_buffer = None
        return h0

    def forward(self, x: Tensor, h_prev: Tensor) -> Tuple[Tensor, Tensor]:
        """
        GRU forward with optional inter-module delays and output delay.
        - x: (B, I)
        - h_prev: (B, H)
        returns:
        y: (B, O)
        h_new: (B, H)
        """
        x = ensure_2d_torch(x, self.input_size, device=self.device, dtype=self.dtype)
        h_prev = ensure_2d_torch(
            h_prev, self.hidden_size, device=self.device, dtype=self.dtype
        )
        B = x.shape[0]

        # --- update hidden buffer: newest previous state at index 0
        if self.max_delay > 0:
            if self.h_buffer is None:
                self.h_buffer = h_prev.unsqueeze(-1).expand(
                    B, self.hidden_size, self.max_delay + 1
                ).contiguous()
            else:
                # concat new state at front, drop oldest
                self.h_buffer = torch.cat(
                    [h_prev.unsqueeze(-1), self.h_buffer[:, :, :-1]], dim=2
                )

        # split weights once
        Wz_in, Wz_hh = self.Wz[:, : self.input_size], self.Wz[:, self.input_size :]
        Wr_in, Wr_hh = self.Wr[:, : self.input_size], self.Wr[:, self.input_size :]
        Wh_in, Wh_hh = self.Wh[:, : self.input_size], self.Wh[:, self.input_size :]

        if self.max_connectivity_delay > 0:
            # delayed path: compute per post-module i with correct (i,j) delays
            h_new = torch.zeros_like(h_prev)

            for i in range(self.num_modules):
                rows = self.module_dims[i]  # indices in post-module i

                # build delayed hidden state specific to post-module i
                h_prev_delayed_i = torch.zeros_like(h_prev)
                for j in range(self.num_modules):
                    cols_j = self.module_dims[j]
                    d_ij = int(self.connectivity_delay[i, j].item())
                    # h_buffer: (B,H,max_delay+1)
                    h_prev_delayed_i[:, cols_j] = self.h_buffer[:, cols_j, d_ij]

                preact_z_full = (
                    x @ Wz_in.t() + h_prev_delayed_i @ Wz_hh.t() + self.bz
                )  # (B,H)
                preact_r_full = (
                    x @ Wr_in.t() + h_prev_delayed_i @ Wr_hh.t() + self.br
                )  # (B,H)
                z_full = torch.sigmoid(preact_z_full)
                r_full = torch.sigmoid(preact_r_full)

                # candidate for selected rows
                h_tilde_in_rows = (
                    x @ Wh_in[rows].t()
                    + (r_full * h_prev_delayed_i) @ Wh_hh[rows].t()
                    + self.bh[rows]
                )  # (B, |rows|)
                h_tilde_rows = self._activation(h_tilde_in_rows)

                z_rows = z_full[:, rows]  # (B, |rows|)
                h_new[:, rows] = (1.0 - z_rows) * h_prev_delayed_i[
                    :, rows
                ] + z_rows * h_tilde_rows
        else:
            # no-delay fast path
            preact_z = x @ Wz_in.t() + h_prev @ Wz_hh.t() + self.bz
            preact_r = x @ Wr_in.t() + h_prev @ Wr_hh.t() + self.br
            z = torch.sigmoid(preact_z)
            r = torch.sigmoid(preact_r)

            h_tilde_in = x @ Wh_in.t() + (r * h_prev) @ Wh_hh.t() + self.bh
            h_tilde = self._activation(h_tilde_in)
            h_new = (1.0 - z) * h_prev + z * h_tilde

        # optional cancellation pulse
        if (self.cancelation_matrix is not None) and (self.cancel_times is not None):
            if self.counter in self.cancel_times:
                h_new = h_new + h_new @ self.cancelation_matrix

        # output (with optional delay)
        if self.output_delay == 0:
            y = torch.sigmoid(h_new @ self.Y.t() + self.bY)
        else:
            # newest previous state is buffer[:,:,0]; use kth-old => index (output_delay-1)
            k = int(self.output_delay) - 1
            h_for_out = self.h_buffer[:, :, k]
            y = torch.sigmoid(h_for_out @ self.Y.t() + self.bY)

        self.counter += 1
        return y, h_new

    # ---------- tools ----------

    @torch.no_grad()
    def cache_policy(self):
        self.Wr_cached.copy_(self.Wr.data)
        self.Wz_cached.copy_(self.Wz.data)
        self.Wh_cached.copy_(self.Wh.data)
        self.Y_cached.copy_(self.Y.data)

    @torch.no_grad()
    def enforce_dale(self, zero_out: bool = False):
        """
        Enforce Dale on the HxH block of (Wr, Wz, Wh) according to self.unittype_W (+1 exc, -1 inh).
        Keeps input->hidden part unchanged.
        """
        if self.unittype_W is None:
            return

        def _apply(mat: Tensor, mat_cached: Tensor) -> Tensor:
            W_in, W_rec = mat[:, : self.input_size], mat[:, self.input_size :]
            Wc_in, Wc_rec = mat_cached[:, : self.input_size], mat_cached[:, self.input_size :]

            exc_mask = self.unittype_W == 1.0
            inh_mask = self.unittype_W == -1.0

            if zero_out:
                # set inconsistent signs to 0
                mask_exc = (W_rec < 0) & exc_mask
                mask_inh = (W_rec > 0) & inh_mask
                W_rec = W_rec.clone()
                W_rec[mask_exc] = 0.0
                W_rec[mask_inh] = 0.0
            else:
                # overwrite inconsistent signs from cached values, preserving magnitude
                mask_exc = (W_rec < 0) & exc_mask
                mask_inh = (W_rec > 0) & inh_mask
                W_rec = W_rec.clone()
                W_rec[mask_exc] = torch.abs(Wc_rec[mask_exc])
                W_rec[mask_inh] = -torch.abs(Wc_rec[mask_inh])

            return torch.cat([W_in, W_rec], dim=1)

        self.Wr.data = _apply(self.Wr.data, self.Wr_cached)
        self.Wz.data = _apply(self.Wz.data, self.Wz_cached)
        self.Wh.data = _apply(self.Wh.data, self.Wh_cached)
        self.cache_policy()

    def _activation(self, x: Tensor) -> Tensor:
        return self._act(x)


# ============================================================
#                      Smoke test
# ============================================================

if __name__ == "__main__":
    torch.set_printoptions(precision=6, sci_mode=False)

    # ---- Simple PolicyGRU_Torch ----
    print("\n=== PolicyGRU_Torch ===")
    B, I, H, O = 2, 5, 16, 3
    pol = PolicyGRU_Torch(I, H, O, seed=0, device="cpu", dtype=torch.float32)

    h = pol.init_hidden(B)
    x = torch.randn(B, I, dtype=torch.float32)
    u, h = pol(x, h)
    print("u shape:", tuple(u.shape), "| h shape:", tuple(h.shape), "| u[0]:", u[0])

    # ---- ModularPolicyGRU_Torch ----
    print("\n=== ModularPolicyGRU_Torch ===")
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
    connectivity_mask = torch.tensor(
        [
            [0.2, 0.1, 0.0],
            [0.1, 0.2, 0.1],
            [0.0, 0.1, 0.2],
        ],
        dtype=torch.float32,
    )

    # output mask per module (probability that module contributes to outputs)
    output_mask = [1.0, 0.5, 0.2]

    # delays (post x pre)
    connectivity_delay = torch.tensor(
        [
            [0, 1, 0],
            [0, 0, 2],
            [0, 0, 0],
        ],
        dtype=torch.int64,
    )

    net = ModularPolicyGRU_Torch(
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
        device="cpu",
        dtype=torch.float32,
    )

    B = 2
    h = net.init_hidden(B)
    x = torch.randn(B, input_size, dtype=torch.float32)

    for t in range(5):
        y, h = net(x, h)
        print(f"t={t}  y[0]:", y[0])

    print("\nSmoke tests complete ✓")
