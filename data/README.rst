Datasets
========

This directory contains two data sets:

1. A Movielens_ 1M dataset downloaded from the webpage_
of the authors of Bayesian Probabilistic Matrix Factorization (BPMF_) method [1]_.
No side information is considered. The BPMF method is the
state of the art against which we compare our results in the case of
no side information.

2. A ChEMBL_ based dataset consists of a sparse matrix containing IC50_ measurements 
for compound-target pairs and of ECFP_ feature matrix of those chemical compounds.   
These are reference datasets for Macau_ - an implementation_ of BPMF with 
side information (here represented by the ECFP matrix). Again, Macau is the state 
of the art against which we compare our matrix factorization solutions with side 
information (and the corresponding approximative averaging sampler).
   
During the installation, 90%-5%-5% train-validation-test splits of the 
datasets will be automatically performed by ``movielensData.py`` and 
``chemblData.py`` scripts.

.. _webpage: http://www.utstat.toronto.edu/~rsalakhu/BPMF.html
.. _BPMF: http://icml2008.cs.helsinki.fi/papers/icml2008proceedings.pdf
.. _Movielens: http://www.utstat.toronto.edu/~rsalakhu/code_BPMF/moviedata.mat
.. _Macau: https://arxiv.org/abs/1509.04610
.. _implementation: https://github.com/jaak-s/macau
.. _ChEMBL: https://www.ebi.ac.uk/chembl/
.. _IC50: http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-346targets.mm
.. _ECFP: http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-compound-feat.mm
.. _dataset: http://files.grouplens.org/datasets/movielens/ml-1m.zip
.. [1] The equivalence to the original Movielens 1M dataset_ not checked.