# telemetry.py ([5]
def pack_diag(**kwargs):
    # tiny passthrough to keep callsites clean
    return dict(**kwargs)


def merge_diag(*dicts):
    out = {}
    for d in dicts:
        if d is not None:
            out.update(d)
    return out
