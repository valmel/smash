#ifndef MF_H
#define MF_H

#include <Eigen/Eigen>
#include <fstream>
#include <unsupported/Eigen/src/SparseExtra/MarketIO.h>
#include <math.h>
#include <algorithm>
#include <random>
#include <numeric>
#include <array>
#include "linalg.h"
#include "sasgd.h"

//////////////////////////////////////////////////////////////////////////
// The main model of the matrix factorization with/wo row side information
//////////////////////////////////////////////////////////////////////////

// SGD unfortunately rather closely couples model, data and its logic.
// A simple and efficient solution (arguably not so elegant as
// a fully independent (SGD) solver) is to derive models from an abstract
// solver class. The model has to implement the abstract member functions
// of the solver class. This exercise is solver dependent. We however currently
// consider only SGD with momentum with adaptive regularization and
// learning rate. The model handles all the data and latent model
// parameters at one place. This gives the user complete freedom
// to handle issues as he/she wishes.

class MF: public SASGD {
private:
	// control
	bool hasRowSideInfo = false;
	bool normalizeGradients = true;
	// data
	SparseMatrix trainMatrix;
	int maxNumOfRowNonzerosTM = 0;
	SparseMatrix valMatrix;
	SparseMatrix testMatrix;
	SparseMatrix rowSideMatrix;
	SparseMatrix rowSideMatrixT;
	int maxNumOfRowNonzerosSM = 0;
	// some data statistics
	int numOfTrainPairs = 0;
	int numOfValPairs = 0;
	int numOfTestPairs = 0;
	int numOfRows = 0;
	int numOfCols = 0;
	int numOfRowFeats = 0;

	// parameters
	double meanTrainValue = 0;
	DenseMatrix U;
	DenseMatrix V;
	DenseMatrix Us;
	DenseMatrix MUs;
	DenseMatrix rsMUs;
	DenseMatrix U_old;
	DenseMatrix V_old;
	DenseMatrix Us_old;
	DenseMatrix U_glob;
	DenseMatrix V_glob;
	DenseMatrix Us_glob;

	//momentums
	DenseMatrix U_m;
	DenseMatrix V_m;
	DenseMatrix Us_m;

	//derivatives
	DenseMatrix dU;
	DenseMatrix dV;
	DenseMatrix dUs;

	//prediction
	DenseMatrix P;

	//data reordering
	std::vector<int> randomRowPermutation;
	int numOfBatchChunks = 0;
	int numOfBatchChunkRows = 0;

	std::vector<std::vector<std::array<int, 2> > > rowToBatchMap;

	//batching
	Eigen::VectorXi nnzPerBatchRowRSM;
	Eigen::VectorXi bRowsPer;
	Eigen::VectorXi bRows;
	Eigen::VectorXi bCols;
	Eigen::VectorXd bVals;
	Eigen::VectorXd bValsCentered;
	Eigen::VectorXd predictions;
	int numOfBatchRows = 0;
	int numOfBatchPairs = 0;
	int bB = 0;
	int bE = 0;
	int bL = 0;
	SparseMatrix rsMbatch;
	SparseMatrix rsMbatchT;

	//validation
	Eigen::VectorXi vRows;
	Eigen::VectorXi vCols;
	Eigen::VectorXd vVals;
	Eigen::VectorXd predVal;

	//test
	Eigen::VectorXi tRows;
	Eigen::VectorXi tCols;
	Eigen::VectorXd tVals;
	Eigen::VectorXd predTest;

	//gradients
	DenseMatrix gradL_U;
	DenseMatrix gradL_V;
	DenseMatrix gradR_U;
	DenseMatrix gradR_V;
	DenseMatrix gradL_Us;
	DenseMatrix gradR_Us;
	DenseMatrix gradF_U;
	DenseMatrix gradF_V;

	//regularization terms
	Eigen::VectorXd regRows;
	Eigen::VectorXd regCols;
	Eigen::VectorXd regRowFeats;

public:
	MF();
	virtual ~MF();
	void useNormalizedGradients(bool ung = true);
	void fillTrainMatrix(int nrows, int ncols, int nnz, int* rows, int* cols,
			double* values);
	void fillValMatrix(int nrows, int ncols, int nnz, int* rows, int* cols,
			double* values);
	void fillTestMatrix(int nrows, int ncols, int nnz, int* rows, int* cols,
			double* values);
	void fillRowSideMatrix(int nrows, int ncols, int nnz, int* rows, int* cols,
			double* values);
	void loadTrainMatrix(std::string filename);
	void loadValMatrix(std::string filename);
	void loadTestMatrix(std::string filename);
	void loadRowSideMatrix(std::string filename);
	int getNumOfTrainPairs();
	int getNumOfValPairs();
	int getNumOfTestPairs();

	inline DenseMatrix& getU() {
		return U;
	};
	inline DenseMatrix& getV() {
		return V;
	};
	inline DenseMatrix& getUs() {
		return Us;
	};
	DenseMatrix& getPrediction();
	Vector& getTestPrediction();
	Vector& getValPrediction();
	double trainError();
	double validError();
	double testError();
	double testErrorPar(int size, double* vec);
	bool doesHaveRowSideInfo();
private:
	//void parameterInitialization();
	void initializeTraining();
	int getBatchSize(int);
	int getSplitSize(int);
	void buildRowSideMatrixBatch(int batch);
	void prepareBatchSplit(int batch);
	void prepareBatch(int batch);
	void computeRegTerms();
	void computePredictions();
	void computeLikelihoodGrads();
	void computeRegGrads();
	void aggregateGrads();
	void computeStochGrads();
	void updateMomentums();
	void eraseMomentums();
	void oneGradStep();
	void permuteData();
	void predictVal();
	void predictTest();
	void PoldToP();
	void PtoPold();
	void PtoPglob();
	void PglobToP();
	void saveFactors();
};
#endif  // MF_H
