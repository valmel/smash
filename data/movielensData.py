import numpy as np
from operator import itemgetter
import scipy.io as io
from scipy.sparse import coo_matrix
import warnings
warnings.filterwarnings("ignore")

def splitProbeToTestAndValidation():    
  # load data
  data = io.loadmat('moviedata.mat')
  probe_vec = data['probe_vec']

  # divide probe
  probe_idx = np.random.randint(2, size = len(probe_vec))
  val_vec = np.asarray([triple for i, triple in enumerate(probe_vec) if probe_idx[i] == 1])
  test_vec = np.asarray([triple for i, triple in enumerate(probe_vec) if probe_idx[i] == 0])
  
  data['val_vec'] = val_vec
  data['test_vec'] = test_vec
  io.savemat('moviedataVT.mat', data)
    
def convertMatToMM():
  # load data
  data = io.loadmat('moviedataVT.mat')
  probe_vec = data['probe_vec']
  train_vec = data['train_vec']
  val_vec = data['val_vec']
  test_vec = data['test_vec']
  
  # Matlab -> Python 
  train_vec[:, 0] = train_vec[:, 0] - 1 # Python indexes from 0 
  train_vec[:, 1] = train_vec[:, 1] - 1 # Python indexes from 0
  train_vec = np.array(sorted(train_vec, key = itemgetter(0, 1)))
  probe_vec[:, 0] = probe_vec[:, 0] - 1 # Python indexes from 0 
  probe_vec[:, 1] = probe_vec[:, 1] - 1 # Python indexes from 0
  val_vec[:, 0] = val_vec[:, 0] - 1 # Python indexes from 0 
  val_vec[:, 1] = val_vec[:, 1] - 1 # Python indexes from 0
  test_vec[:, 0] = test_vec[:, 0] - 1 # Python indexes from 0 
  test_vec[:, 1] = test_vec[:, 1] - 1 # Python indexes from 0
  
  shapex = np.max(train_vec[:, 0]) + 1
  shapey = np.max(train_vec[:, 1]) + 1 
  
  train =  coo_matrix((train_vec[:, 2], (train_vec[:, 0], train_vec[:, 1])), shape = (shapex, shapey))
  val =  coo_matrix((val_vec[:, 2], (val_vec[:, 0], val_vec[:, 1])), shape = (shapex, shapey))
  test =  coo_matrix((test_vec[:, 2], (test_vec[:, 0], test_vec[:, 1])), shape = (shapex, shapey))
  probe = coo_matrix((probe_vec[:, 2], (probe_vec[:, 0], probe_vec[:, 1])), shape = (shapex, shapey))
  
  io.mmwrite('movielens_train', train, field = 'integer')
  io.mmwrite('movielens_val', val, field = 'integer')
  io.mmwrite('movielens_test', test, field = 'integer')
  io.mmwrite('movielens_probe', probe, field = 'integer')
      
if __name__ == '__main__':
  splitProbeToTestAndValidation()
  convertMatToMM()