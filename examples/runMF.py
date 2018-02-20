import smash
import scipy.io as io
import numpy as np

#####################
# make a choice
#####################

#dataset = 'movielens'
dataset = 'chembl'
#normalizeGradients = False 
normalizeGradients = True
#sideInfo = False
sideInfo = True

#####################
# end of your choice
#####################

# we currently have no side info for movielens
if dataset == 'movielens':
  sideInfo = False
  
mf = smash.PyMF()  

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
  rowSideMat = io.mmread("../data/chembl-IC50-compound-feat.mm")
  mf.fillRowSideMatrix(rowSideMat)

# parameters common for the datasets  
mf.setVerbosity(1)
mf.setAlphaDecay(1.5)
mf.setLRDecay(1.0)
mf.setInitializationScalingFactor(0.3)
mf.setMinimalNumOfEpochs(3)
mf.setMomentum(0.8)
mf.setNumOfBatches(9)
mf.setMaxEpoch(100)
mf.setLatentDim(100)
mf.useNormalizedGradients(normalizeGradients)

# parameter which differ for the datasets
if (dataset == 'movielens'):
  mf.setInitialAlpha(0.32) # initial regularization parameter
  if normalizeGradients:
    mf.setInitialLR(320000.)
  else:
    mf.setInitialLR(32.)
  
if (dataset == 'chembl'):
  mf.setInitialAlpha(0.32) # initial regularization parameter
  if normalizeGradients:
    mf.setInitialLR(3200.)
  else:                    
    mf.setInitialLR(10.)

mf.train()

meanTrainValue = trainMat.sum() / len(trainMat.row)

U = mf.getU()
V = mf.getV()
P = mf.getPrediction()
P_here = meanTrainValue + U.dot(V.transpose())
if (dataset == 'chembl' and sideInfo == True):
  Us = mf.getUs()
  P_here += (rowSideMat*Us).dot(V.transpose())
    
print("norm(P-P_here) = ", np.linalg.norm(P-P_here))

P_test = mf.getTestPrediction()
P_test_here = meanTrainValue + np.sum(U[testMat.row, :]*V[testMat.col, :], 1)
if (dataset == 'chembl' and sideInfo == True):
  rowSideMatCSR = rowSideMat.tocsr()
  Us = mf.getUs()
  P_test_here += np.sum((rowSideMatCSR[testMat.row, :]*Us)*V[testMat.col, :], 1)
print("norm(P_test-P_test_here) = ", np.linalg.norm(P_test-P_test_here))

P_val = mf.getValPrediction()
P_val_here = meanTrainValue + np.sum((U[valMat.row, :])*V[valMat.col, :], 1)
if (dataset == 'chembl' and sideInfo == True):
  Us = mf.getUs()
  P_val_here += np.sum((rowSideMatCSR[valMat.row, :]*Us)*V[valMat.col, :], 1)
print("norm(P_val-P_val_here) = ", np.linalg.norm(P_val-P_val_here))