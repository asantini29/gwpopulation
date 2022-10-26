TEST_BACKENDS = ["numpy", "jax.numpy"]
try:
    import cupy  # noqa

    TEST_BACKENDS.append("cupy")
except ImportError:
    pass
