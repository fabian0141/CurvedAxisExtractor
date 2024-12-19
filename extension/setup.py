from setuptools import setup, Extension
import numpy

module = Extension('contour', sources=['init.c', 'pointlist.c', 'contour.c', 'splitcontour.c', 'findcorner.c', 
                                       'line.c', 'point.c', 'segment.c', 'circles.c', 'circle.c', 'circle2/angle.c'], 
                                       include_dirs=[numpy.get_include(), './'], extra_compile_args=['-g'])

setup(
    name='contour',
    version='1.0',
    description='Python Package with C extension',
    ext_modules=[module],
)