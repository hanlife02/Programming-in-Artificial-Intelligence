import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
__version__ = '0.0.1'
sources = ['src/Tensor.h']
setup(
    name='mytensor',
    version=__version__,
    author='Ethan',
    author_email='ethan@hanlife02.com',
    packages=find_packages(),
    zip_safe=False,
    install_requires=['torch'],
    python_requires='>=3.8',
    license='MIT',
    ext_modules=[
        CUDAExtension(
        name='mytensor',
        sources=sources)
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    classifiers=[
       'License :: OSI Approved :: MIT License',
    ],
)