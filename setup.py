from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="Fluid Simulation",
    ext_modules=cythonize("sim.pyx"),
    include_dirs=[np.get_include()]
)
