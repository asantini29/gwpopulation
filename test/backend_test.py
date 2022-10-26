import numpy
import pytest

import gwpopulation


def test_unsupported_backend_raises_value_error():
    with pytest.raises(ValueError):
        gwpopulation.set_backend("fail")


def test_set_backend_numpy():
    gwpopulation.set_backend("numpy")
    from gwpopulation.utils import xp

    assert xp == numpy


def test_set_backend_jax():
    pytest.importorskip("jax.numpy")
    import jax.numpy as jnp

    gwpopulation.set_backend("jax.numpy")
    from gwpopulation.utils import xp

    assert jnp == xp


def test_enable_cupy_deprecated():
    with pytest.deprecated_call():
        gwpopulation.enable_cupy()


def test_disable_cupy_deprecated():
    with pytest.deprecated_call():
        gwpopulation.disable_cupy()
