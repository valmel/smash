#!/usr/bin/env python
import numpy as np
import scipy.io as io
import os as os
from smash.sampling import HPS
from smash import PyMF
import warnings
warnings.filterwarnings("ignore")

####################################
####################################
## SET UP MODEL TO SAMPLE FROM
####################################
####################################

##################
# make a choice
##################

#dataset = 'movielens'
dataset = 'chembl'
#normalizeGradients = False
normalizeGradients = True
#sideInfo = False
sideInfo = True

##################
# end of your choice
##################


# we currently have no side info for movielens
if dataset == 'movielens':
  sideInfo = False

mf = PyMF()

if (dataset == 'movielens'):
  trainMat = io.mmread("../data/movielens_train.mtx")
  valMat = io.mmread("../data/movielens_val.mtx")
  testMat = io.mmread("../data/movielens_test.mtx")
  
if (dataset == 'chembl'):  
  trainMat = io.mmread("../data/chembl-IC50-346targets_train.mtx")
  valMat = io.mmread("../data/chembl-IC50-346targets_val.mtx")
  testMat = io.mmread("../data/chembl-IC50-346targets_test.mtx")

mf.fillTrainMatrix(trainMat)
mf.fillValMatrix(valMat)
mf.fillTestMatrix(testMat)

if (dataset == 'chembl' and sideInfo == True):  
  rowSideMat = io.mmread("../data/chembl-IC50-compound-feat.mm") # ecfp (+ ...)
  mf.fillRowSideMatrix(rowSideMat)

# parameters common for the datasets
if normalizeGradients == True:
  mf.useNormalizedGradients(True)
mf.setVerbosity(0)
mf.setAlphaDecay(1.5)
mf.setLRDecay(1.0)
mf.setInitializationScalingFactor(0.3)
mf.setMinimalNumOfEpochs(3) # quasi constant
mf.setMomentum(0.8) # quasi constant
mf.setMaxEpoch(2000)
mf.setNumOfBatches(9) # quasi constant 
mf.setLatentDim(1)

if (dataset == 'movielens'):
  mf.setInitialAlpha(0.32) # initial regularization parameter
  # here is normalization no so important as for chembl (no side info)
  if normalizeGradients:
    mf.setInitialLR(320000.)
  else:
    mf.setInitialLR(320.)
  
if (dataset == 'chembl'):
  mf.setInitialAlpha(0.32) # initial regularization parameter
  # no normalization leads to a significantly worse RMSE
  if normalizeGradients:
    mf.setInitialLR(3200.)
  else:                    
    mf.setInitialLR(10.)

####################################
####################################
## SET UP HYPEPPARAMETER SAMPLING
####################################
####################################

case = 0

# define hyperparameter ranges for sampling
ranges = []
# only SGD with fixed alphas and LRs
if case == 0: 
  rcoord = ['alpha', 'LR', 'L']
  ranges.append(np.asarray([0.32])) # range of alpha
  ranges.append(np.asarray([3200.])) # range of LR
  ranges.append(np.asarray([10])) # range of L
# variate both alpha and the learning rate with speeds (F2F)  
if case == 1:
  rcoord = ['alpha', 'LR', 'L', 'dalpha', 'dLR']  
  ranges.append(np.asarray([0.04, 0.08, 0.16])) # range of alpha
  ranges.append(np.asarray([40., 80., 160., 320.])) # range of LR
  ranges.append(np.asarray([10, 20, 40, 80, 160])) # range of L
  ranges.append(np.asarray([1.2, 1.6, 2.0])) # speed of alpha decrease
  ranges.append(np.asarray([1.0, 1.5, 2.0])) # range of LR decrease
if case == 2:
  rcoord = ['alpha', 'LR', 'L', 'dalpha', 'dLR']  
  ranges.append(np.asarray([0.04, 0.08])) # range of alpha
  ranges.append(np.asarray([40., 80., 160., 320.])) # range of LR
  ranges.append(np.asarray([1])) # range of L
  ranges.append(np.asarray([1.6, 2.0])) # speed of alpha decrease
  ranges.append(np.asarray([1.0])) # range of LR decrease

  
hps = HPS()
hps.setModel(mf)
hps.setRanges(rcoord, ranges)
pwd = os.path.dirname(os.path.realpath(__file__))
hps.setDataDir(pwd + '/../data')
hps.setOutputDir(pwd + '/output')
  
hps.sample()
hps.saveResults('results')

  