from setuptools import setup, Extension
import numpy

module = Extension('contour', sources=['contour.c'], include_dirs=[numpy.get_include()])

setup(
    name='contour',
    version='1.0',
    description='Python Package with C extension',
    ext_modules=[module],
)