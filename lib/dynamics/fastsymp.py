# fastsymp.py

import os
import time
import sympy as sp
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# -------------------------
# Configuration knobs
# -------------------------
SIMPLIFY_LEVEL = os.getenv("ZROBOTICS_SIMPLIFY", "light")  # "none" | "light" | "full"
PARALLEL_THRESHOLD = int(
    os.getenv("ZROBOTICS_PAR_THRESH", "64")
)  # entries needed to justify process pool
LOG_SIMPLIFY = os.getenv("ZROBOTICS_SIMPLIFY_LOG", "0") == "1"


def _log(msg: str) -> None:
    if LOG_SIMPLIFY:
        print(msg)


def pinv_lr(J):
    """Return a light-weight symbolic pseudoinverse for a SymPy Matrix J.
    Uses left or right form depending on shape, assuming full rank.
    Falls back to J.pinv() if (J.T*J) or (J*J.T) is singular at build time.
    """
    try:
        if J.rows >= J.cols:
            # left pseudoinverse: (J^T J)^(-1) J^T
            return (J.T * J).inv() * J.T
        else:
            # right pseudoinverse: J^T (J J^T)^(-1)
            return J.T * (J * J.T).inv()
    except Exception:
        # symbolic singularity → safe fallback
        return J.pinv()


def _entrywise_map(func, entries, max_workers=None):
    """Map `func` over a flat list of entries.
    Parallelize only if there are enough entries to amortize process overhead.
    """
    n = len(entries)
    if n >= PARALLEL_THRESHOLD:
        workers = max_workers or os.cpu_count() or 1
        if workers > 1:
            with ProcessPoolExecutor(workers) as pool:
                return list(pool.map(func, entries))
    # Fallback to serial map
    return [func(e) for e in entries]


# -------------------------
# Lightweight "tidy" pass
# -------------------------
def tidy(expr: sp.Basic, level: str = SIMPLIFY_LEVEL) -> sp.Basic:
    """Cheap, controllable simplifier.
    - Removes conjugates (assume real symbols are intended)
    - On scalars: together → cancel → trigsimp (and optional radsimp/powsimp for 'full')
    - On matrices: apply entrywise
    """
    if level == "none":
        return expr

    expr = expr.replace(sp.conjugate, lambda x: x)

    if isinstance(expr, sp.MatrixBase):
        flat = list(expr)
        flat = [tidy(e, level) for e in flat]
        return expr._new(expr.rows, expr.cols, flat)

    # Scalar / Expr path
    e = sp.together(expr)
    e = sp.cancel(e)
    # e = sp.trigsimp(e)
    if level == "full":
        e = sp.radsimp(e)
        e = sp.powsimp(e, force=True)
    return e


# -------------------------
# Heavier, but still optimized simplifier (legacy compatible)
# -------------------------
def fast_simplify1(expr: sp.Basic, max_workers: int = None) -> sp.Basic:
    """Legacy 'heavier' simplifier (kept for compatibility), made safer/faster:
    - Avoid unconditional parallelism (thresholded)
    - Strip conjugates once
    - Use CSE; apply rational/trig/cancel/factor/radsimp on reduced parts
    """
    t0 = time.perf_counter()
    expr = expr.replace(sp.conjugate, lambda x: x)

    # MATRIX path
    if isinstance(expr, sp.MatrixBase):
        flat = list(expr)
        # Map recursively, parallel only if big enough
        mapped = _entrywise_map(
            lambda e: fast_simplify1(e, max_workers=max_workers), flat, max_workers
        )
        M = expr._new(expr.rows, expr.cols, mapped)
        out = sp.trigsimp(M)
        dt = time.perf_counter() - t0
        _log(f"Simplifier=fast_simplify1 | matrix {expr.shape} | time={dt:.3f}s")
        return out

    # SCALAR path
    reps, reduced = sp.cse(expr, optimizations="basic")
    cleaned = []
    for sub in reduced:
        s = sp.ratsimp(sub)
        s = sp.trigsimp(s)
        s = sp.cancel(s)
        s = sp.factor(s)
        s = sp.radsimp(s)
        # Mild local cleanup on sums/products
        if s.is_Add:
            s = sp.Add(
                *(sp.cancel(sp.trigsimp(t)) for t in s.as_ordered_terms()),
                evaluate=True,
            )
        elif s.is_Mul:
            s = sp.Mul(
                *(sp.cancel(sp.trigsimp(f)) for f in s.as_ordered_factors()),
                evaluate=True,
            )
        cleaned.append(s)

    out = cleaned[0] if len(cleaned) == 1 else sp.Add(*cleaned, evaluate=True)
    if reps:
        # xreplace is faster than chained subs
        out = out.xreplace(dict(reps))

    dt = time.perf_counter() - t0
    _log(f"Simplifier=fast_simplify1 | scalar | time={dt:.3f}s")
    return out


# -------------------------
# Default lightweight fast simplifier (recommended)
# -------------------------
def fast_simplify(expr: sp.Basic, max_workers: int = None) -> sp.Basic:
    """Recommended fast simplifier:
    - Very light passes (together → cancel → trigsimp) + CSE
    - Entrywise for matrices with thresholded parallelism
    - No evalf/nsimplify to avoid expensive rational reconstruction
    """
    t0 = time.perf_counter()
    expr = expr.replace(sp.conjugate, lambda x: x)

    # MATRIX path (entry-wise only)
    if isinstance(expr, sp.MatrixBase):
        flat = list(expr)
        simplifier = partial(fast_simplify, max_workers=max_workers)
        mapped = _entrywise_map(simplifier, flat, max_workers)
        result = sp.trigsimp(expr._new(expr.rows, expr.cols, mapped))
        dt = time.perf_counter() - t0
        _log(f"Simplifier=fast_simplify   | matrix {expr.shape} | time={dt:.3f}s")
        return result

    # SCALAR path
    reps, reduced = sp.cse(expr, optimizations="basic")
    cleaned = []
    for sub in reduced:
        s = sp.together(sub)
        s = sp.cancel(s)
        s = sp.trigsimp(s)
        cleaned.append(s)

    out = cleaned[0] if len(cleaned) == 1 else sp.Add(*cleaned, evaluate=True)
    if reps:
        out = out.xreplace(dict(reps))

    dt = time.perf_counter() - t0
    _log(f"Simplifier=fast_simplify   | scalar | time={dt:.3f}s")
    return out
