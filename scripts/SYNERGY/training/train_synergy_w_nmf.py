#!/usr/bin/env python3
import os
import argparse
import numpy as np

# --- NMF (multiplicative updates) ---

def nmf_mu(V, K, max_iter=500, tol=1e-5, seed=0):
    """
    Solve min ||V - W H||_F^2  s.t. W>=0,H>=0 using multiplicative updates.
    V: (m, T) nonnegative
    W: (m, K)
    H: (K, T)
    """
    rng = np.random.default_rng(seed)
    m, T = V.shape
    W = rng.random((m, K)) + 1e-3
    H = rng.random((K, T)) + 1e-3
    eps = 1e-12

    prev = None
    for it in range(max_iter):
        # H <- H * (W^T V) / (W^T W H)
        H *= (W.T @ V) / (W.T @ W @ H + eps)
        # W <- W * (V H^T) / (W H H^T)
        W *= (V @ H.T) / (W @ (H @ H.T) + eps)

        # normalize columns of W (common in synergy work)
        W /= (np.linalg.norm(W, axis=0, keepdims=True) + eps)

        err = np.linalg.norm(V - W @ H, ord="fro")
        if prev is not None and abs(prev - err) < tol:
            break
        prev = err

    return W, H, prev

def vaf(V, W, H):
    R = V - W @ H
    return 1.0 - (np.sum(R*R) / (np.sum(V*V) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act_npz", type=str, required=True,
                    help="NPZ containing activation matrix V with key 'V' (shape 6xT) or key 'act' (Tx6).")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--out", type=str, default="synergy/saved_model/W_model.npz")
    ap.add_argument("--max_iter", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    z = np.load(args.act_npz, allow_pickle=True)
    if "V" in z:
        V = np.asarray(z["V"], dtype=float)
        if V.shape[0] != 6:
            raise ValueError(f"Expected V shape (6,T). Got {V.shape}.")
    elif "act" in z:
        A = np.asarray(z["act"], dtype=float)
        # allow either (T,6) or (6,T)
        if A.shape[-1] == 6:
            V = A.T
        elif A.shape[0] == 6:
            V = A
        else:
            raise ValueError(f"Can't interpret 'act' shape {A.shape} as 6-muscle activations.")
    else:
        raise KeyError("NPZ must contain key 'V' or 'act'.")

    # Ensure nonnegativity and remove min-activation bias if present
    V = np.clip(V, 0.0, 1.0)

    W, H, err = nmf_mu(V, args.K, max_iter=args.max_iter, seed=args.seed)
    score = vaf(V, W, H)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, W=W, H=H, VAF=score, err=err, K=args.K)

    print(f"[NMF] saved: {args.out}")
    print(f"[NMF] K={args.K}, VAF={score:.4f}, final_err={err:.6f}")

if __name__ == "__main__":
    main()

