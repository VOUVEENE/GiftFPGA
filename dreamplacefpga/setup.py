from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'gift_adj_cpp',
        ['gift_adj_matrix.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++11', '-O3'],
    ),
]

setup(
    name='gift_adj_cpp',
    ext_modules=ext_modules,
    zip_safe=False,
)
