import smash
import scipy.io as io

trainMat = io.mmread("../data/chembl-IC50-346targets_train.mtx")
valMat = io.mmread("../data/chembl-IC50-346targets_val.mtx")
testMat = io.mmread("../data/chembl-IC50-346targets_test.mtx")
rowSideMat = io.mmread("../data/chembl-IC50-compound-feat.mm")

result = smash.mf(trainMat, valMat, testMat,
                  rowSide = rowSideMat,
                  verbosity = 1,
                  alpha = 0.32,
                  dalpha = 1.5,
                  num_latent = 100,
                  ilr = 3200.)