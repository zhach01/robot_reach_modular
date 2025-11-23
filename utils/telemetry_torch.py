# telemetry_torch.py
# -*- coding: utf-8 -*-
"""
Torch-friendly telemetry helpers.

This is a direct port of utils/telemetry.py to the Torch side.
There is no NumPy / backend dependency: everything is backend-agnostic and
can be used from both the NumPy and Torch codepaths.

Functions
---------
- pack_diag(**kwargs)
- merge_diag(*dicts)
"""


def pack_diag(**kwargs):
    """
    Tiny passthrough to keep callsites clean.

    Examples
    --------
    diag = pack_diag(
        q=q,
        qd=qd,
        tau=tau,
        F=F,
    )
    """
    return dict(**kwargs)


def merge_diag(*dicts):
    """
    Merge multiple diagnostic dictionaries.

    Later dictionaries override earlier ones (like dict.update).

    Parameters
    ----------
    *dicts : dict or None
        Any number of dictionaries or None.

    Returns
    -------
    out : dict
        Flat dictionary containing all keys/values from non-None inputs.
    """
    out = {}
    for d in dicts:
        if d is not None:
            out.update(d)
    return out


# --------------------------------------------------------------------------- #
# Tiny smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("[telemetry_torch] Simple smoke test...")

    d1 = pack_diag(a=1, b=2)
    d2 = pack_diag(b=3, c=4)
    merged = merge_diag(d1, None, d2)

    print("  d1:", d1)
    print("  d2:", d2)
    print("  merged:", merged)

    print("[telemetry_torch] smoke ✓")
