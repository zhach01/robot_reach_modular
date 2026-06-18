import numpy as np


def matrix_sqrt_spd(S, eps=1e-12):
    Ssym = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(Ssym)
    w = np.maximum(w, eps)
    return V @ np.diag(np.sqrt(w)) @ V.T


def matrix_isqrt_spd(S, eps=1e-12):
    Ssym = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(Ssym)
    w = np.maximum(w, eps)
    return V @ np.diag(1.0 / np.sqrt(w)) @ V.T


def matrix_sqrt_isqrt_spd(S, eps=1e-12):
    """Return (S^{1/2}, S^{-1/2}) for SPD S from a single eigendecomposition.

    Bit-identical to (matrix_sqrt_spd(S), matrix_isqrt_spd(S)) but avoids the
    second eigh when a caller needs both (e.g. the operational-space Kv gain).
    """
    Ssym = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(Ssym)
    w = np.maximum(w, eps)
    sqrt_w = np.sqrt(w)
    return V @ np.diag(sqrt_w) @ V.T, V @ np.diag(1.0 / sqrt_w) @ V.T


def right_pinv_rows(A, eps=1e-8):
    G = A @ A.T
    return A.T @ np.linalg.inv(G + eps * np.eye(G.shape[0]))


def right_pinv_rows_weighted(A, Rw, eps=1e-8):
    G = A @ Rw @ A.T
    return Rw @ A.T @ np.linalg.inv(G + eps * np.eye(G.shape[0]))
