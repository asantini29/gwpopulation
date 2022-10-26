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


def set_backend(backend="numpy", verbose=True):
    global __backend__
    global xp
    if backend == __backend__:
        return
    supported = ["numpy", "cupy", "jax.numpy"]
    scipy_module = {"numpy": "scipy", "cupy": "cupyx.scipy", "jax.numpy": "jax.scipy"}
    if backend not in supported:
        raise ValueError(
            f"Backed {backend} not supported, should be in {', '.join(supported)}"
        )
    import importlib

    try:
        xp = importlib.import_module(backend)
        scipy = importlib.import_module(scipy_module[backend])
    except ImportError:
        if verbose:
            print(f"Cannot import {backend}, falling back to numpy")
        set_backend(backend="numpy", verbose=False)
        return
    for module in __all_with_xp:
        if verbose:
            print(f"Setting backend to {backend}")
        __backend__ = backend
        module.__backend__ = backend
        module.xp = xp
        module.betaln = scipy.special.betaln
        module.erf = scipy.special.erf
        module.gammaln = scipy.special.gammaln
        module.i0e = scipy.special.i0e


set_backend("cupy", verbose=False)
