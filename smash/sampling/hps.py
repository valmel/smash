from mpi4py import MPI
import numpy as np
import pandas as pd
import os as os 
from numpy.random import randint

class HPS:
  def __init__(self):
    self.COMM = MPI.COMM_WORLD
    self.size = self.COMM.Get_size()
    self.rank = self.COMM.Get_rank()
      
  def setModel(self, model):
    self.mf = model
    self.mf.setVerbosity(0)

  def chunks(self, l, n):
    k, m = len(l) // n, len(l) % n
    return [l[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

  def setRanges(self, rcoords, ranges):
    self.rcoords = rcoords
    self.ranges = ranges
    self.results = []
    
  def setDataDir(self, ddir):
    self.ddir = ddir
    if self.rank == 0 and not os.path.isdir(self.ddir):
      raise RuntimeError('Data directory does not exist...')
        
  def setOutputDir(self, odir):
    self.odir = odir
    self.factors_dir = self.odir + 'factors/'

    if not os.path.isdir(self.odir):
      os.mkdir(self.odir)
    if not os.path.isdir(self.odir):
      raise RuntimeError('Output directory does not exist and its creation fails...')
    else:
      if not os.path.isdir(self.factors_dir):
        os.mkdir(self.factors_dir)

  def saveFactors(self, i):
    self.mf.saveFactors(self.factors_dir, str(i))

  def learnMF(self, i, indxs):  
    for j in range(len(self.ranges)):
      if self.rcoords[j] == 'alpha':
        self.mf.setInitialAlpha(self.ranges[j][indxs[j]])
      if self.rcoords[j] == 'LR':
        self.mf.setInitialLR(self.ranges[j][indxs[j]])
      if self.rcoords[j] == 'L':
        self.mf.setLatentDim(self.ranges[j][indxs[j]])
      if self.rcoords[j] == 'dalpha':
        self.mf.setAlphaDecay(self.ranges[j][indxs[j]])  
      if self.rcoords[j] == 'dLR':
        self.mf.setLRDecay(self.ranges[j][indxs[j]])
          
    self.mf.train()
    
    # now collect the statistics
    coords = []
    for j in range(len(self.ranges)):
      coords.append(self.ranges[j][indxs[j]])
    result = [str(i)] + coords + [self.mf.trainError(), self.mf.validError(), self.mf.testError(), \
        self.mf.getEpoch(), self.mf.getDuration(), self.mf.getAlpha()]
    print(result)
    self.results.append(result)
      
  def sample(self):
    indxs = np.zeros((len(self.ranges),), dtype = np.int)
      
    self.nsamples = self.size # for simplicity 
    if self.rank == 0:
      jobs = list(range(self.nsamples))
      jobs = self.chunks(jobs, self.COMM.size)
    else:
      jobs = None
      
    #Scatter jobs across cores.
    jobs = self.COMM.scatter(jobs, root = 0)
    
    self.results = []
    self.pred = np.zeros(self.mf.getNumOfTestPairs())
    for i in jobs:
      print("rank " + str(self.COMM.rank) + ": " + str(i) + " job")
      for j in range(len(self.ranges)):
        indxs[j] = randint(0, len(self.ranges[j]))
      self.learnMF(i, indxs)
      self.saveFactors(i)
      self.pred += self.mf.getTestPrediction()
  
    self.results = MPI.COMM_WORLD.gather(self.results, root = 0)
             
    if(len(jobs) > 0):
      self.pred /= len(jobs) #the average prediction of each worker
    self.meanPred = np.zeros(len(self.pred))
    MPI.COMM_WORLD.Reduce([self.pred, MPI.DOUBLE], [self.meanPred, MPI.DOUBLE], op = MPI.SUM, root = 0)
    if self.rank == 0:
      print("size =", self.size)
      print("Mean test error = " + str(self.mf.testErrorPar(self.meanPred/self.size)))
    print("rank " + str(self.COMM.rank) + " finished.")
    
  def saveResults(self, fname):
    # results to dataframe
    if self.rank == 0:
      self.results = [_i for temp in self.results for _i in temp]
      self.results = np.asarray(self.results)
      self.df = pd.DataFrame(self.results)
      output_columns = ['train_error', 'val_error', 'test_error', 'epoch', 'duration', 'final_alpha']
      columns = self.rcoords + output_columns
      columns = ['index'] + self.rcoords + output_columns
      self.df.columns = columns
      self.df.set_index(['index'])
      self.df.to_csv(self.odir + '/' + fname, index = False)  
      #print(self.df)
      #print(self.df.columns)
      # convert some floats to int
      #self.df['index'] = self.df['index'].apply(np.int64)
      #self.df['L'] = self.df['L'].apply(np.int64)
      #self.df['epoch'] = self.df['epoch'].apply(np.int64)