Examples
========

1. ```python runMF.py``` allows you to run a single instance of incomplete matrix 
factorization for **Movielens** or **ChEMBL** dataset (or whatever you supply). 
The script is an example how to use the OOP interface to the underlying C++ matrix 
factorization class. This is a good starting point if you want to extend 
the interface and/or the underlying implementation. For a description of the 
datasets, see  `../data/README.rst`_

2. ```python runMFfunc.py``` allows you to run a single instance of incomplete matrix 
for **ChEMBL** dataset (or whatever you supply). Instead of the OOP interface 
to the underlying C++ matrix factorization class, it employs a single function interface 
of the implementation.
 
3. ``mpirun -np n python runSMASH.py`` allows you to run MC sampling in 
hyperparameter space as defined in the file. Here ``n`` is number of independent 
MPI processes.

.. _`../data/README.rst`: ../data/README.rst
