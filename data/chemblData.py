import scipy.io as io
import random as random
import numpy as np
from scipy.sparse import coo_matrix
import warnings
warnings.filterwarnings("ignore")


def makeTrainValTest():
  ic50 = io.mmread('chembl-IC50-346targets.mm')
  
  prob_train = 0.9
  prob_val = 0.05
  #prob_test = 1. - prob_train - prob_val
  
  choice = []
  for index in range(len(ic50.data)):
    rn = random.random()
    if rn < prob_train:
      choice.append(0)
    else:
      if rn < prob_train + prob_val:
        choice.append(1)
      else:
        choice.append(2)
        
  train_row = np.asarray([r for i, r in enumerate(ic50.row) if choice[i] == 0])
  train_col = np.asarray([c for i, c in enumerate(ic50.col) if choice[i] == 0])
  train_data = np.asarray([d for i, d in enumerate(ic50.data) if choice[i] == 0])
  
  val_row = np.asarray([r for i, r in enumerate(ic50.row) if choice[i] == 1])
  val_col = np.asarray([c for i, c in enumerate(ic50.col) if choice[i] == 1])
  val_data = np.asarray([d for i, d in enumerate(ic50.data) if choice[i] == 1])
  
  test_row = np.asarray([r for i, r in enumerate(ic50.row) if choice[i] == 2])
  test_col = np.asarray([c for i, c in enumerate(ic50.col) if choice[i] == 2])
  test_data = np.asarray([d for i, d in enumerate(ic50.data) if choice[i] == 2])
  
  train =  coo_matrix((train_data, (train_row, train_col)), shape=ic50.shape)
  val =  coo_matrix((val_data, (val_row, val_col)), shape=ic50.shape)
  test =  coo_matrix((test_data, (test_row, test_col)), shape=ic50.shape)
  
  io.mmwrite('chembl-IC50-346targets_train', train)
  io.mmwrite('chembl-IC50-346targets_val', val)
  io.mmwrite('chembl-IC50-346targets_test', test)
  
if __name__ == '__main__':
  makeTrainValTest()