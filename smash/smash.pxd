from libcpp cimport bool
from libcpp.string cimport string

cdef extern from "Eigen/Dense" namespace "Eigen":
    cdef cppclass MatrixXd:
        MatrixXd()
        MatrixXd(int nrow, int ncol)
        int rows()
        int cols()
        double* data()
    cdef cppclass VectorXd:
        VectorXd()
        VectorXd(int n)
        int size()
        double* data()
    T Map[T](double* x, int nrows, int ncols)

cdef extern from "mf.h":
  cdef cppclass MF:
    MF() except +
    #MF(int num_latent)
    void fillTrainMatrix(int nrows, int ncols, int N, int* rows, int* cols, double* values)
    void fillValMatrix(int nrows, int ncols, int N, int* rows, int* cols, double* values)
    void fillTestMatrix(int nrows, int ncols, int N, int* rows, int* cols, double* values)
    void fillRowSideMatrix(int nrows, int ncols, int N, int* rows, int* cols, double* values)
    int getNumOfTrainPairs()
    int getNumOfValPairs()
    int getNumOfTestPairs()
    void setVerbosity(int v)
    void setLatentDim(int ld)
    void setInitialAlpha(double alpha)
    void setAlphaDecay(double dalpha)
    void setInitialLR(double ilr)
    void setLRDecay(double lrd)
    void useNormalizedGradients(bool ung)
    void setInitializationScalingFactor(double d)
    void setMomentum(double m)
    void setMaxEpoch(int me)
    void setNumOfBatches(int nb)
    void setMinimalNumOfEpochs(int mne)
    void train()
    double trainError()
    double validError()
    double testError()
    double testErrorPar(int size, double* vec)
    int getEpoch()
    double getAlpha()
    double getLR()
    double getDuration()
    MatrixXd getU();
    MatrixXd getV();
    MatrixXd getUs();
    MatrixXd getPrediction();
    VectorXd getValPrediction();
    VectorXd getTestPrediction();
    bool doesHaveRowSideInfo();