import numpy as np


# optional shared helpers
def blend(a, b, t):
    # t in [0,1]; returns (1-t)*a + t*b
    return (1.0 - t) * a + t * b


def clip01(x):
    return float(np.clip(x, 0.0, 1.0))
