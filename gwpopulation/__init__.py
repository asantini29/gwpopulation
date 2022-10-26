"""
GWPopulation
============

A collection of code for doing population inference.

All of this code will run on either CPUs or GPUs using cupy for GPU
acceleration.

This includes:
  - commonly used likelihood functions in the Bilby framework.
  - population models for gravitational-wave sources.
  - selection functions for gravitational-wave sources.

The code is hosted at `<www.github.com/ColmTalbot/gwpopulation>`_.
"""
from . import conversions, hyperpe, models, utils, vt
from .hyperpe import RateLikelihood

try:
    from ._version import version as __version__
except ModuleNotFoundError:  # development mode
    __version__ = "unknown"


__all_with_xp = [
    hyperpe,
    models.mass,
    models.redshift,
    models.spin,
    utils,
    vt,
]
__backend__ = ""
_backends = ["numpy", "cupy", "jax.numpy"]
_scipy_module = {"numpy": "scipy", "cupy": "cupyx.scipy", "jax.numpy": "jax.scipy"}


def disable_cupy():
    from warnings import warn

    warn(
        f"Function enable_cupy is deprecated, use set_backed('cupy') instead",
        DeprecationWarning,
    )
    set_backend(backend="numpy")


def enable_cupy():
    from warnings import warn

    warn(
        f"Function enable_cupy is deprecated, use set_backed('cupy') instead",
        DeprecationWarning,
    )
    set_backend(backend="cupy")


def set_backend(backend="numpy"):
    global __backend__
    if backend not in _backends:
        raise ValueError(
            f"Backed {backend} not supported, should be in {', '.join(_backends)}"
        )
    elif backend == __backend__:
        return

    import importlib

    xp = importlib.import_module(backend)
    scs = importlib.import_module(_scipy_module[backend]).special
    for module in __all_with_xp:
        __backend__ = backend
        module.__backend__ = backend
        module.xp = xp
    utils.scs = scs
    models.mass.scs = scs


set_backend("numpy")
