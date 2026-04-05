"""
Minimal chumpy compatibility shim for Python 3.12+.

The original SMPL .pkl files were serialised with chumpy arrays.
This stub converts them to plain numpy arrays on load so that
smplx can read the model without requiring the real chumpy package.
"""

import numpy as np


class Ch(np.ndarray):
    """
    Drop-in stub for chumpy.Ch.
    Behaves like a numpy array so smplx can use it directly.
    """

    def __new__(cls, x=None, *args, **kwargs):
        if x is None:
            arr = np.array([])
        elif hasattr(x, '__array__'):
            arr = np.asarray(x)
        else:
            try:
                arr = np.array(x)
            except Exception:
                arr = np.array([])
        return arr.view(cls)

    # smplx / SMPL code sometimes accesses .r to get the numpy value
    @property
    def r(self):
        return np.asarray(self)

    def __reduce__(self):
        return (Ch, (np.asarray(self),))

    # Silence attribute errors for chumpy-specific methods
    def __getattr__(self, name):
        raise AttributeError(name)


def array(x, *args, **kwargs):
    return np.array(x)


# Aliases used by some SMPL pkl files
zeros  = np.zeros
ones   = np.ones
arange = np.arange


class reordering_csc_matrix:
    """Stub for sparse matrix type used in some SMPL variants."""
    def __init__(self, *args, **kwargs):
        pass
