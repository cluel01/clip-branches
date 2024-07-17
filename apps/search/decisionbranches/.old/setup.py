from setuptools import find_packages, setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np


extensions = [Extension('decisionbranches.cython.utils', ["decisionbranches/cython/utils.pyx"]),
            Extension('decisionbranches.cython.functions', ["decisionbranches/cython/functions.pyx"],
                      include_dirs=[np.get_include()],extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp']),
            Extension('py_kdtree.cython.float32.box_query', ["py_kdtree/cython/float32/box_query.pyx"],
            include_dirs=[np.get_include()],extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp']),
              Extension('py_kdtree.cython.float32.point_query', ["py_kdtree/cython/float32/point_query.pyx"],
            include_dirs=[np.get_include()],extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']),
        Extension('py_kdtree.cython.float64.box_query', ["py_kdtree/cython/float64/box_query.pyx"],
            include_dirs=[np.get_include()],extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp']),
              Extension('py_kdtree.cython.float64.point_query', ["py_kdtree/cython/float64/point_query.pyx"],
            include_dirs=[np.get_include()],extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'])]


setup(
    name='decisionbranches',
    packages=find_packages(),
    version='0.1.0',
    description='Code for paper: Rapid Rare-Object Search via Decision Branches',
    author='anonymous',
    license='',
    ext_modules=cythonize(extensions),
    zip_safe=False
)
