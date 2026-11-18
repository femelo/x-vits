"""Builds the optional Cython extension for monotonic alignment search.

Without Cython or a C compiler the package still installs: ``piper_train``
falls back to an equivalent numba implementation.
"""
from setuptools import setup

ext_modules = []
try:
    import numpy
    from Cython.Build import cythonize
    from setuptools import Extension

    ext_modules = cythonize(
        [Extension("piper_train.vits.monotonic_align.core",
                   ["src/piper_train/vits/monotonic_align/core.pyx"],
                   include_dirs=[numpy.get_include()])],
        language_level=3,
    )
except Exception as e:  # pragma: no cover
    print(f"lilt: skipping the Cython MAS extension ({e}); the numba fallback will be used")

setup(ext_modules=ext_modules)
