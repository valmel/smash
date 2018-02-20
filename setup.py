#!/usr/bin/env python
import os, sys
import numpy as np
from glob import glob

from distutils.command.build_clib import build_clib
from distutils.errors    import DistutilsSetupError
from distutils.sysconfig import get_python_inc
#from distutils.core import setup

from setuptools import setup
from setuptools import Extension
from setuptools.command.install import install

from Cython.Distutils import build_ext
from Cython.Build import cythonize

import tempfile
import tarfile
import shutil

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve

def downloadEigen():
    dest = "deps/eigen3.3.4"
    url  = "http://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2"
    dirName = "eigen-eigen-5a0156e40feb"
    if os.path.isdir(dest + "/Eigen"):
        return
    print("Downloading Eigen (v3.3)...")
    tmpdir = tempfile.mkdtemp()
    bzfile = tmpdir + "/eigen.tar.bz2"
    urlretrieve(url, bzfile)
    print("Download complete. Extracting Eigen ...")
    tf = tarfile.open(bzfile, "r:bz2")
    if not os.path.exists(dest):
        os.makedirs(dest)
    tf.extractall(path = tmpdir)
    print("Extracting complete.")
    tmpeigen = tmpdir + "/" + dirName
    shutil.move(tmpeigen + "/Eigen", dest)
    shutil.move(tmpeigen + "/unsupported", dest)
    shutil.rmtree(tmpdir)

def postInstallCMD():
    from subprocess import call
    dir = os.path.dirname(os.path.realpath(__file__))
    print(dir)
    call([sys.executable, dir + '/data/movielensData.py'],
         cwd = dir + '/data')
    call([sys.executable, dir + '/data/chemblData.py'],
         cwd = dir + '/data')

class postInstall(install):
    """Post-installation for installation mode."""
    def run(self):
      install.run(self)
      self.execute(postInstallCMD, (), msg="Running post install task")
      
ldirs = ["/usr/local/lib"]      
includes = ['smash/cpp', 'deps/eigen3.3.4',  np.get_include(), get_python_inc(), "/usr/local/include"]
      
exts = [Extension("smash.smash",
    sources = ["smash/smash.pyx"] + list(filter(lambda a: a.find("tests.cpp") < 0,
                               glob('smash/cpp/*.cpp'))),
    include_dirs = includes,
    library_dirs = ldirs,
    runtime_library_dirs = ldirs,
    extra_compile_args = ['-fopenmp', '-static', '-O3', '-fstrict-aliasing', '-DNDEBUG', '-Wall', '-std=c++11'],
    extra_link_args = ['-fopenmp', '-lstdc++'],
    language = "c++")]

def main():
  downloadEigen()
  setup(
    name = 'smash',
    version = '0.1.0',
    packages = ['smash', 'smash.sampling'],
    description = 'SMASH: Sampling with Monotone Annealing based Stochastic Homotopy',
    long_description = 'Implements a simulated annealing based SGD solver in C++.'
                      +' The homotopy in the regularization parameters facilitates'
                      +' convexification of the problem. The found solutions are close'
                      +' to the global one(s). They represent samples from the posterior'
                      +' around its modes.',                   
    url = 'https://github.com/valmel/smash',
    author = 'Valdemar Melicher',
    author_email = 'Valdemar.Melicher@UAntwerpen.be',
    license = 'MIT',
    classifiers = [
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research',
      'Topic :: Scientific/Engineering :: Mathematics',
      'Topic :: Scientific/Engineering :: Incomplete Matrix Factorization',
      'Topic :: Scientific/Engineering :: Collaborative Filtering', 
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python',
      'Programming Language :: Python :: 2.7',
      'Programming Language :: Python :: 3',
    ],
    keywords = 'MF SGD SA simulated annealing convexification sampling',      
    ext_modules = cythonize(exts, include_path = sys.path),
    install_requires = ['numpy', 'scipy', 'pandas', 'mpi4py'],
    package_data = {
      'examples': ['examples/runMF.py', 'examples/runSMASH.py'],
    },
    cmdclass={
      'build_ext': build_ext,
      'install': postInstall,
      },
  )

if __name__ == '__main__':
    main()