cimport cython
import numpy as np
cimport numpy as np
import scipy as sp
np.import_array()
import scipy.io as io

cdef matview(MatrixXd *A):
  cdef int nrow = A.rows()
  cdef int ncol = A.cols()
  if nrow == 0:
    return np.zeros( (nrow, ncol) )
  cdef np.double_t[:,:] view = <np.double_t[:nrow:1, :ncol]> A.data()
  return np.asarray(view)

cdef vecview(VectorXd *v):
  cdef int size = v.size()
  if size == 0:
    return np.zeros( 0 )
  cdef np.double_t[:] view = <np.double_t[:size]> v.data()
  return np.asarray(view)

# passive checking. It is up to the user to supply the right format
def checkTVTmatrices(trainMat, valMat, testMat):
  if type(trainMat) != sp.sparse.coo.coo_matrix:
    raise TypeError("trainMat must be coo_matrix from scipy.sparse")
  if np.any(np.isnan(trainMat.data)):
    raise ValueError("trainMat may not contain NaNs")
  if type(valMat) != sp.sparse.coo.coo_matrix:
    raise TypeError("valMat must be coo_matrix from scipy.sparse")
  if np.any(np.isnan(valMat.data)):
    raise ValueError("valMat may not contain NaNs")
  if type(testMat) != sp.sparse.coo.coo_matrix:
    raise TypeError("testMat must be coo_matrix from scipy.sparse")
  if np.any(np.isnan(testMat.data)):
    raise ValueError("testMat may not contain NaNs")
    
# passive checking. It is up to the user to supply the right format    
def checkRowSideInfo(side):
  if side != None and type(side) != sp.sparse.coo.coo_matrix:
    raise TypeError("rowSide is specified but it is not coo_matrix from scipy.sparse")
  if side != None and np.any(np.isnan(side.data)):
    raise ValueError("rowSide may not contain NaNs")    

def mf(trainMat,
       valMat,
       testMat,
       rowSide = None,
       verbosity = 1,
       alpha = 0.2,
       dalpha = 1.5,
       num_latent = 10,
       ilr = 1000.):
  
    checkTVTmatrices(trainMat, valMat, testMat)
    checkRowSideInfo(rowSide)
    
    # create mf object
    cdef int D = np.int32(num_latent)
    cdef MF *mf = new MF()
    
    # fill train data
    cdef np.ndarray[int] rows = trainMat.row.astype(np.int32, copy = False)
    cdef np.ndarray[int] cols = trainMat.col.astype(np.int32, copy = False)
    cdef np.ndarray[np.double_t] values = trainMat.data.astype(np.double, copy = False)
    mf.fillTrainMatrix(trainMat.shape[0], trainMat.shape[1], trainMat.getnnz(), &rows[0], &cols[0], &values[0])
    
    # fill validation data
    cdef np.ndarray[int] vrows = valMat.row.astype(np.int32, copy = False)
    cdef np.ndarray[int] vcols = valMat.col.astype(np.int32, copy = False)
    cdef np.ndarray[np.double_t] vvalues = valMat.data.astype(np.double, copy = False)
    mf.fillValMatrix(valMat.shape[0], valMat.shape[1], valMat.getnnz(), &vrows[0], &vcols[0], &vvalues[0])
    
    # fill test data
    cdef np.ndarray[int] trows = testMat.row.astype(np.int32, copy = False)
    cdef np.ndarray[int] tcols = testMat.col.astype(np.int32, copy = False)
    cdef np.ndarray[np.double_t] tvalues = testMat.data.astype(np.double, copy = False)
    mf.fillTestMatrix(testMat.shape[0], testMat.shape[1], testMat.getnnz(), &trows[0], &tcols[0], &tvalues[0])
      
    ## side information
    cdef np.ndarray[int] rsrows
    cdef np.ndarray[int] rscols
    cdef np.ndarray[np.double_t] rsvalues
    if rowSide != None:
      rsrows = rowSide.row.astype(np.int32, copy = False)
      rscols = rowSide.col.astype(np.int32, copy = False)
      rsvalues = rowSide.data.astype(np.double, copy = False)
      mf.fillRowSideMatrix(rowSide.shape[0], rowSide.shape[1], rowSide.getnnz(), &rsrows[0], &rscols[0], &rsvalues[0])
      
    mf.setVerbosity(verbosity)
    mf.setLatentDim(num_latent)
    mf.setInitialAlpha(alpha)
    mf.setAlphaDecay(dalpha)
    mf.setInitialLR(ilr)

    mf.train()
    
    cdef MatrixXd U_raw = mf.getU()
    cdef MatrixXd V_raw  = mf.getV()
    cdef MatrixXd Us_raw = mf.getUs()

    cdef np.ndarray[np.double_t, ndim = 2] U = matview( & U_raw ).copy()
    cdef np.ndarray[np.double_t, ndim = 2] V = matview( & V_raw ).copy()
    cdef np.ndarray[np.double_t, ndim = 2] Us = matview( & Us_raw ).copy()

    del mf

    return U, V, Us
    
cdef class PyMF:
    cdef MF* _mf
    def __cinit__(self):
        self._mf = new MF()   
    def fillTrainMatrix(self, mat):
        if type(mat) != sp.sparse.coo.coo_matrix:
            raise TypeError("fillTrainMatrix: matrix must be coo_matrix from scipy.sparse")
        if np.any(np.isnan(mat.data)):
            raise ValueError("fillTrainMatrix: matrix may not contain NaNs")
        cdef np.ndarray[int] rows = mat.row.astype(np.int32, copy = False)
        cdef np.ndarray[int] cols = mat.col.astype(np.int32, copy = False)
        cdef np.ndarray[np.double_t] values = mat.data.astype(np.double, copy = False)
        self._mf.fillTrainMatrix(mat.shape[0], mat.shape[1], mat.getnnz(), &rows[0], &cols[0], &values[0])
    def fillValMatrix(self, mat):
        if type(mat) != sp.sparse.coo.coo_matrix:
            raise TypeError("fillValMatrix: matrix must be coo_matrix from scipy.sparse")
        if np.any(np.isnan(mat.data)):
            raise ValueError("fillValMatrix: matrix may not contain NaNs")
        cdef np.ndarray[int] rows = mat.row.astype(np.int32, copy = False)
        cdef np.ndarray[int] cols = mat.col.astype(np.int32, copy = False)
        cdef np.ndarray[np.double_t] values = mat.data.astype(np.double, copy = False)
        self._mf.fillValMatrix(mat.shape[0], mat.shape[1], mat.getnnz(), &rows[0], &cols[0], &values[0])
    def fillTestMatrix(self, mat):
        if type(mat) != sp.sparse.coo.coo_matrix:
            raise TypeError("fillTestMatrix: matrix must be coo_matrix from scipy.sparse")
        if np.any(np.isnan(mat.data)):
            raise ValueError("fillTestMatrix: matrix may not contain NaNs")
        cdef np.ndarray[int] rows = mat.row.astype(np.int32, copy = False)
        cdef np.ndarray[int] cols = mat.col.astype(np.int32, copy = False)
        cdef np.ndarray[np.double_t] values = mat.data.astype(np.double, copy = False)
        self._mf.fillTestMatrix(mat.shape[0], mat.shape[1], mat.getnnz(), &rows[0], &cols[0], &values[0])
    def fillRowSideMatrix(self, mat):
        if type(mat) != sp.sparse.coo.coo_matrix:
            raise TypeError("fillRowSideMatrix: matrix must be coo_matrix from scipy.sparse")
        if np.any(np.isnan(mat.data)):
            raise ValueError("fillRowSideMatrix: matrix may not contain NaNs")
        cdef np.ndarray[int] rows = mat.row.astype(np.int32, copy = False)
        cdef np.ndarray[int] cols = mat.col.astype(np.int32, copy = False)
        cdef np.ndarray[np.double_t] values = mat.data.astype(np.double, copy = False)
        if mat != None:
            self._mf.fillRowSideMatrix(mat.shape[0], mat.shape[1], mat.getnnz(), &rows[0], &cols[0], &values[0])
    def getNumOfTrainPairs(self):
        return self._mf.getNumOfTrainPairs()
    def getNumOfValPairs(self):
        return self._mf.getNumOfValPairs()
    def getNumOfTestPairs(self):
        return self._mf.getNumOfTestPairs()
    def setVerbosity(self, v):
        self._mf.setVerbosity(v)
    def setLatentDim(self, ld):
        self._mf.setLatentDim(ld)
    def setInitialAlpha(self, alpha):
        self._mf.setInitialAlpha(alpha)
    def setAlphaDecay(self, dalpha):
        self._mf.setAlphaDecay(dalpha)
    def setInitialLR(self, ilr):
        self._mf.setInitialLR(ilr)
    def setLRDecay(self, lrd):
        self._mf.setLRDecay(lrd)    
    def useNormalizedGradients(self, ung):
        self._mf.useNormalizedGradients(ung)
    def setInitializationScalingFactor(self, isf):
        self._mf.setInitializationScalingFactor(isf)
    def setMomentum(self, m):
        self._mf.setMomentum(m)
    def setMaxEpoch(self, me):
        self._mf.setMaxEpoch(me)
    def setNumOfBatches(self, nb):
        self._mf.setNumOfBatches(nb)
    def setMinimalNumOfEpochs(self, mne):
        self._mf.setMinimalNumOfEpochs(mne)        
    def train(self):
        self._mf.train()
    def trainError(self):
        return self._mf.trainError()
    def validError(self):
        return self._mf.validError()
    def testError(self):
        return self._mf.testError()
    def testErrorPar(self, np.ndarray[np.double_t] vec):
        return self._mf.testErrorPar(len(vec), &vec[0])
    def getEpoch(self):
        return self._mf.getEpoch()
    def getAlpha(self):
        return self._mf.getAlpha()
    def getLR(self):
        return self._mf.getLR()
    def getDuration(self):
        return self._mf.getDuration()
    def getU(self):
        cdef MatrixXd U_raw = self._mf.getU()
        cdef np.ndarray[np.double_t, ndim = 2] U = matview( & U_raw ).copy()
        return U;
    def getV(self):
        cdef MatrixXd V_raw  = self._mf.getV()
        cdef np.ndarray[np.double_t, ndim = 2] V = matview( & V_raw ).copy()
        return V;
    def getUs(self):
        cdef MatrixXd Us_raw = self._mf.getUs()
        cdef np.ndarray[np.double_t, ndim = 2] Us = matview( & Us_raw ).copy()
        return Us
    def getPrediction(self):
        cdef MatrixXd p_raw = self._mf.getPrediction()
        cdef np.ndarray[np.double_t, ndim = 2] P = matview( & p_raw ).copy()
        return P
    def getValPrediction(self):
        cdef VectorXd p_raw = self._mf.getValPrediction()
        cdef np.ndarray[np.double_t] p = vecview( & p_raw ).copy()
        return p
    def getTestPrediction(self):
        cdef VectorXd p_raw = self._mf.getTestPrediction()
        cdef np.ndarray[np.double_t] p = vecview( & p_raw ).copy()
        return p
    def saveFactors(self, fdir, index):
        cdef nameU = fdir + 'U_' + index
        cdef nameV = fdir + 'V_' + index
        cdef nameUs = fdir + 'Us_' + index
        io.mmwrite(nameU, self.getU())
        io.mmwrite(nameV, self.getV())
        if self._mf.doesHaveRowSideInfo():
            io.mmwrite(nameUs, self.getUs())
    def __dealloc__(self):
        del self._mf