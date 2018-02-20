SMASH: Sampling with Monotone Annealing based Stochastic Homotopy (for incomplete matrix factorization)
=======================================================================================================

This repository contains C++ implementation of SMASH together with
corresponding python wrappers. For the corresponding pure Python implementation 
(or alternatively a bit of Cython), see https://github.com/valmel/smashpy. 
The initial commits of both repositories have very similar functionality, 
but a divergence is to be expected. The reimplementation of the code in C++ was 
necessary to allow for scalability to bigger datasets (>~ 1 GB).

The contribution is few-folded:

First, a "simulated annealing" based SGD solver is implemented. Since 
our ultimate criterion is simplicity, we start from SGD with momentum.
We introduce continuation for the regularization parameter. This homotopy 
facilitates a convexification of the iterative process. The found local 
minima of the regularized cost functional (loss) are consequently close 
to the global one(s). Moreover, the convexified problem is more isotropic 
which leads to a rather quick convergence.

The initial regularization parameter is set above the optimal level 
of regularization (can be much higher). The optimal regularization 
is found automatically during the run of the method. In principle, 
no expensive hyperparameter search (or cross-validation) is necessary.

Further, we argue that the resulting SASGD produces samples around modes of 
the posterior. We employ this property and introduce Monte Carlo (MC) 
(and are planning Quasi Monte Carlo (QMC)) sampling in hyperparameter space. 
The resulting embarrassingly parallel approximative sampler (except the prediction phase)
leads to accuracy comparable to fully Bayesian samplers.  

Installation
------------
 
 Run in the main folder:

```python setup.py install --user```

Examples
--------

Examples are located in ``examples`` directory. See the corresponding 
``examples/README.rst`` file there
