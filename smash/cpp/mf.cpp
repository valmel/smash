#include "mf.h"

MF::MF() :
		SASGD() {
	numOfBatchChunks = 100; //depends on architecture (not optimized!)
}

MF::~MF() {
}

void MF::useNormalizedGradients(bool ung) {
	normalizeGradients = ung;
}

void MF::fillTrainMatrix(int nrows, int ncols, int nnz, int* rows, int* cols,
		double* values) {
	if (verbosity > 2)
		std::cout << "fillTrainMatrix(()" << std::endl;
	ScipyCooToEigenSparse(nrows, ncols, nnz, rows, cols, values, trainMatrix);
	//numTrainPairs = trainMatrix.nonZeros();
	numOfTrainPairs = nnz;
	meanTrainValue = trainMatrix.sum() / trainMatrix.nonZeros();
	numOfRows = nrows;
	numOfCols = ncols;
	randomRowPermutation.resize(numOfRows);
}

void MF::fillValMatrix(int nrows, int ncols, int nnz, int* rows, int* cols,
		double* values) {
	if (verbosity > 2)
		std::cout << "fillValMatrix()" << std::endl;
	vRows.resize(nnz);
	vCols.resize(nnz);
	vVals.resize(nnz);
	vRows = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(rows, nnz);
	vCols = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(cols, nnz);
	vVals = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(values, nnz);
	ScipyCooToEigenSparse(nrows, ncols, nnz, rows, cols, values, valMatrix);
	//numValPairs = valMatrix.nonZeros();
	numOfValPairs = nnz;
}

void MF::fillTestMatrix(int nrows, int ncols, int nnz, int* rows, int* cols,
		double* values) {
	if (verbosity > 2)
		std::cout << "fillTestMatrix()" << std::endl;
	tRows.resize(nnz);
	tCols.resize(nnz);
	tVals.resize(nnz);
	tRows = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(rows, nnz);
	tCols = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(cols, nnz);
	tVals = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(values, nnz);
	ScipyCooToEigenSparse(nrows, ncols, nnz, rows, cols, values, testMatrix);
	//numTestPairs = testMatrix.nonZeros();
	numOfTestPairs = nnz;
}

void MF::fillRowSideMatrix(int nrows, int ncols, int nnz, int* rows, int* cols,
		double* values) {
	if (verbosity > 2)
		std::cout << "fillRowSideMatrix()" << std::endl;
	ScipyCooToEigenSparse(nrows, ncols, nnz, rows, cols, values, rowSideMatrix);
	numOfRowFeats = ncols;
	rowSideMatrixT = rowSideMatrix.transpose();
	hasRowSideInfo = true;

}

void MF::loadTrainMatrix(std::string filename) {
	if (verbosity > 2)
		std::cout << "loadTrainMatrix()" << std::endl;
	loadMarket(trainMatrix, filename);
	numOfTrainPairs = trainMatrix.nonZeros();
	meanTrainValue = trainMatrix.sum() / trainMatrix.nonZeros();
	maxNumOfRowNonzerosTM = getMaxNumberOfRowNonzeros(trainMatrix);
	numOfRows = trainMatrix.rows();
	numOfCols = trainMatrix.cols();
	randomRowPermutation.resize(numOfRows);
}

void MF::loadValMatrix(std::string filename) {
	if (verbosity > 2)
		std::cout << "loadValMatrix()" << std::endl;
	loadMarket(valMatrix, filename);
	//numOfValPairs = valMatrix.nonZeros();
	SparseMatrixToCoo(valMatrix, numOfValPairs, vRows, vCols, vVals);
}

void MF::loadTestMatrix(std::string filename) {
	if (verbosity > 2)
		std::cout << "loadTestMatrix()" << std::endl;
	loadMarket(testMatrix, filename);
	//numOfTestPairs = testMatrix.nonZeros();
	SparseMatrixToCoo(testMatrix, numOfTestPairs, tRows, tCols, tVals);

}

void MF::loadRowSideMatrix(std::string filename) {
	if (verbosity > 2)
		std::cout << "loadRowSideMatrix()" << std::endl;
	loadMarket(rowSideMatrix, filename);
	numOfRowFeats = rowSideMatrix.cols();
	maxNumOfRowNonzerosSM = getMaxNumberOfRowNonzeros(rowSideMatrix);
	rowSideMatrixT = rowSideMatrix.transpose();
	hasRowSideInfo = true;
}

int MF::getNumOfTrainPairs(){
	return numOfTrainPairs;
}

int MF::getNumOfValPairs(){
	return numOfValPairs;
}

int MF::getNumOfTestPairs(){
	return numOfTestPairs;
}

void MF::initializeTraining() {
	if (verbosity > 2)
		std::cout << "initializeTraining()" << std::endl;
	numOfBatchRows = ceil(((double) numOfRows) / ((double) numOfBatches));
	numOfBatchChunkRows = ceil(
			((double) numOfRows) / ((double) numOfBatchChunks));

	U = isf / sqrt(1.0 * latentDim)
			* Eigen::MatrixXd::Random(numOfRows, latentDim);
	V = isf / sqrt(1.0 * latentDim)
			* Eigen::MatrixXd::Random(numOfCols, latentDim);

	U_m = Eigen::MatrixXd::Zero(numOfRows, latentDim);
	V_m = Eigen::MatrixXd::Zero(numOfCols, latentDim);

	dU = Eigen::MatrixXd::Zero(numOfRows, latentDim);
	dV = Eigen::MatrixXd::Zero(numOfCols, latentDim);

	predVal.resize(numOfValPairs);
	predTest.resize(numOfTestPairs);

	if (hasRowSideInfo) {
		Us = Eigen::MatrixXd::Zero(numOfRowFeats, latentDim);
		dUs = Eigen::MatrixXd::Zero(numOfRowFeats, latentDim);
		rsMUs = Eigen::MatrixXd::Zero(numOfRows, latentDim);
		rsMUs = rowSideMatrix * Us;
		gradR_Us = Eigen::MatrixXd::Zero(numOfRowFeats, latentDim);
		Us_m = Eigen::MatrixXd::Zero(numOfRowFeats, latentDim);
		MUs = Eigen::MatrixXd::Zero(numOfBatchRows, latentDim);
	}
}

inline int MF::getBatchSize(int n) {
	if (n < 0 || n > numOfBatches) {
		std::cout << "getBatchSize: check n!" << std::endl;
		exit(0);
	}
	int bB = n * numOfBatchRows;
	int bE = std::min((n + 1) * numOfBatchRows, numOfRows);
	return bE - bB;
}

inline int MF::getSplitSize(int n) {
	if (n < 0 || n > numOfBatchChunks) {
		std::cout << "getSplitSize: check n!" << std::endl;
		exit(0);
	}
	int sB = n * numOfBatchChunkRows;
	int sE = std::min((n + 1) * numOfBatchChunkRows, numOfRows);
	return sE - sB;
}

void MF::permuteData() {
	if (verbosity > 2)
		std::cout << "permuteData()" << std::endl;
	auto engine = std::default_random_engine { };
	engine.seed(unsigned(time(NULL)));
	std::iota(randomRowPermutation.begin(), randomRowPermutation.end(), 0); // aka range(len(randomRowPermutation))
	std::shuffle(std::begin(randomRowPermutation),
			std::end(randomRowPermutation), engine);
}

void MF::prepareBatchSplit(int batch) {
	//erase the old struct
	for (unsigned int s = 0; s < rowToBatchMap.size(); s++) {
		rowToBatchMap[s].clear();
	}
	rowToBatchMap.clear();

	//resize the struct;
	rowToBatchMap.resize(numOfBatchChunks);

	//fill the split struct
	int chunk;
	std::array<int, 2> rowToBatch;
	for (int i = 0; i < numOfBatchPairs; i++) {
		chunk = floor(bRows[i] / numOfBatchChunkRows);
		rowToBatch[0] = i;
		rowToBatch[1] = bRows[i];
		rowToBatchMap[chunk].push_back(rowToBatch);
	}
}

void MF::buildRowSideMatrixBatch(int batch) {
	// a smarter memory aware version of the following
	//for (int i = 0; i < numOfBatchPairs; i++)
	//  rsMbatch.row(i) = rowSideMatrix.row(bRows[i]);

	nnzPerBatchRowRSM = Eigen::VectorXi::Zero(numOfBatchPairs);
	for (int i = 0; i < numOfBatchPairs; i++) {
		nnzPerBatchRowRSM[i] = rowSideMatrix.outerIndexPtr()[bRows[i] + 1]
				- rowSideMatrix.outerIndexPtr()[bRows[i]];
	}

	// compute the memory which needs to be reserved
	int nnz = 0;
	for (int i = 0; i < numOfBatchPairs; i++)
		nnz += nnzPerBatchRowRSM[i];
	rsMbatch.reserve(nnz);

	// fill in outerIndexPtr of rsMbatch
	int idx = 0;
	for (int i = 0; i < numOfBatchPairs; i++) {
		rsMbatch.outerIndexPtr()[i] = idx;
		idx += nnzPerBatchRowRSM[i];
	}
	rsMbatch.outerIndexPtr()[numOfBatchPairs] = idx;

	int bRow = 0;
	int rsmRow = 0;
	int startOfRow = 0;
	for (unsigned int s = 0; s < rowToBatchMap.size(); s++)
		for (unsigned int p = 0; p < rowToBatchMap[s].size(); p++) {
			bRow = rowToBatchMap[s][p][0];
			rsmRow = rowToBatchMap[s][p][1];
			startOfRow = rsMbatch.outerIndexPtr()[bRow];
			idx = 0;
			for (int i = rowSideMatrix.outerIndexPtr()[rsmRow];
					i < rowSideMatrix.outerIndexPtr()[rsmRow + 1]; ++i) {
				rsMbatch.innerIndexPtr()[startOfRow + idx] =
						rowSideMatrix.innerIndexPtr()[i];
				rsMbatch.valuePtr()[startOfRow + idx] =
						rowSideMatrix.valuePtr()[i];
				idx++;
			}
		}
}

void MF::prepareBatch(int batch) {
	if (verbosity > 2)
		std::cout << "prepareBatch()" << std::endl;
	bB = batch * numOfBatchRows;
	bE = std::min((batch + 1) * numOfBatchRows, numOfRows);
	bL = bE - bB;

	int oldNumOfBatchPairs = numOfBatchPairs;
	SparseMatrixRowRangePer(bB, bE, randomRowPermutation, trainMatrix,
			numOfBatchPairs, bRowsPer, bCols, bVals);

	if (numOfBatchPairs > oldNumOfBatchPairs) {
		bRows.resize(numOfBatchPairs);
		gradL_U = Eigen::MatrixXd::Zero(numOfBatchPairs, latentDim);
		gradL_V = Eigen::MatrixXd::Zero(numOfBatchPairs, latentDim);
		gradR_U = Eigen::MatrixXd::Zero(numOfBatchPairs, latentDim);
		gradR_V = Eigen::MatrixXd::Zero(numOfBatchPairs, latentDim);
		if (hasRowSideInfo) {
			rsMbatch = SparseMatrix(numOfBatchPairs, numOfRowFeats);
			rsMbatchT = SparseMatrix(numOfRowFeats, numOfBatchPairs);
			rsMbatch.reserve(numOfBatchPairs * maxNumOfRowNonzerosSM);
			gradL_Us = Eigen::MatrixXd::Zero(numOfRowFeats, latentDim);
		}
	} else {
		gradL_U.fill(0.);
		gradL_V.fill(0.);
		gradR_U.fill(0.);
		gradR_V.fill(0.);
		if (hasRowSideInfo) {
			rsMbatch.setZero();
			gradL_Us.fill(0.);
		}
	}

	for (int i = 0; i < numOfBatchPairs; i++)
		bRows[i] = randomRowPermutation[bRowsPer[i]]; // bRows is indexing the latent variables
	prepareBatchSplit(batch);

	if (hasRowSideInfo) {
		//rsMbatch.resize(0,0);
		//rsMbatch.resize(numOfBatchPairs, numOfRowFeats);
		buildRowSideMatrixBatch(batch);
		rsMbatchT = rsMbatch.transpose();
		//MUs.fill(0.);
	}
}

void MF::computeRegTerms() {
	if (verbosity > 2)
		std::cout << "computeRegTerms()" << std::endl;
	//regRows = np.sum(self.U[self.bRows,:]**2, 1)
	regRows = U.rowwise().squaredNorm();
	//regCols = np.sum(self.V[self.bCols,:]**2, 1)
	regCols = V.rowwise().squaredNorm();
	if (hasRowSideInfo)
		regRowFeats = Us.rowwise().squaredNorm();
}

void MF::computePredictions() {
	if (verbosity > 2)
		std::cout << "computePredictions()" << std::endl;
	bValsCentered.resize(numOfBatchPairs);
	bValsCentered = bVals.array() - meanTrainValue;
	predictions.resize(numOfBatchPairs);
	for (int p = 0; p < numOfBatchPairs; p++) {
		predictions[p] = V.row(bCols[p]).dot(U.row(bRows[p]));
	}
	if (hasRowSideInfo) {
		for (int p = 0; p < numOfBatchPairs; p++)
			predictions[p] += V.row(bCols[p]).dot(rsMUs.row(bRows[p]));
	}

	double fid = (predictions - bValsCentered).norm();
	fid = fid * fid;
	double f = fid + 0.5 * alpha * (regRows.sum() + regCols.sum());
	if (hasRowSideInfo)
		f += 0.5 * alpha * regRowFeats.sum();
}

void MF::computeLikelihoodGrads() {
	if (verbosity > 2)
		std::cout << "computeLikelihoodGrads()" << std::endl;
	double mult;

	for (int p = 0; p < numOfBatchPairs; p++) {
		mult = 2. * (predictions[p] - bValsCentered[p]);
		gradL_U.row(p) = mult * V.row(bCols[p]);
		gradL_V.row(p) = mult * U.row(bRows[p]);
	}
	if (hasRowSideInfo) {
		gradL_Us.noalias() = rsMbatchT * gradL_U;
	}
}

void MF::computeRegGrads() {
	if (verbosity > 2)
		std::cout << "computeRegGrads()" << std::endl;
	for (int p = 0; p < numOfBatchPairs; p++) {
		gradR_U.row(p) = U.row(bRows[p]);
		gradR_V.row(p) = V.row(bCols[p]);
	}
	if (hasRowSideInfo)
		gradR_Us = Us;
}

void MF::aggregateGrads() {
	if (verbosity > 2)
		std::cout << "aggregateGrads()" << std::endl;
	for (int p = 0; p < numOfBatchPairs; p++) {
		dU.row(bRows[p]) += gradF_U.row(p);
		dV.row(bCols[p]) += gradF_V.row(p);
	}
}

void MF::computeStochGrads() {
	if (verbosity > 2)
		std::cout << "computeStochGrads()" << std::endl;
	computeLikelihoodGrads();
	computeRegGrads();

	gradF_U = gradL_U + alpha * gradR_U;
	gradF_V = gradL_V + alpha * gradR_V;

	dU.fill(0.);
	dV.fill(0.);

	aggregateGrads();

	if (hasRowSideInfo)
		dUs = gradL_Us + alpha * gradR_Us;
}

void MF::updateMomentums() {
	if (verbosity > 2)
		std::cout << "updateMomentums()" << std::endl;
	if (normalizeGradients) {
		dU /= dU.norm();
		dV /= dV.norm();
	}

	U_m = momentum * U_m + LR * dU / numOfBatchPairs;
	V_m = momentum * V_m + LR * dV / numOfBatchPairs;

	if (hasRowSideInfo) {
		if (normalizeGradients)
			dUs /= dUs.norm();
		Us_m = momentum * Us_m + LR * dUs / numOfBatchPairs;
	}
}

void MF::oneGradStep() {
	if (verbosity > 2)
		std::cout << "oneGradStep()" << std::endl;
	U -= U_m;
	V -= V_m;
	if (hasRowSideInfo)
		Us -= Us_m;
	rsMUs = rowSideMatrix * Us;
	//cout << Us.max();
}

double MF::trainError() {
	if (verbosity > 2)
		std::cout << "trainError()" << std::endl;
	// uses the last batch only
	int batch = numOfBatches - 1;
	prepareBatch(batch);
	computeRegTerms();
	computePredictions();
	double fid = (predictions - bValsCentered).norm();
	fid = fid * fid;
	return sqrt(fid / numOfBatchPairs);
}

void MF::predictVal() {
	if (verbosity > 2)
		std::cout << "predictVal()" << std::endl;
	for (int p = 0; p < numOfValPairs; p++) {
		predVal[p] = V.row(vCols[p]).dot(U.row(vRows[p]));
	}
	predVal = predVal.array() + meanTrainValue;

	if (hasRowSideInfo) {
		rsMUs = rowSideMatrix * Us;
		for (int p = 0; p < numOfValPairs; p++)
			predVal[p] += V.row(vCols[p]).dot(rsMUs.row(vRows[p]));
	}
}

double MF::validError() {
	if (verbosity > 2)
		std::cout << "validError()" << std::endl;
	predictVal();
	double fid = (predVal - vVals).norm();
	return sqrt(fid * fid / numOfValPairs);
}

void MF::predictTest() {
	if (verbosity > 2)
		std::cout << "predictTest()" << std::endl;
	for (int p = 0; p < numOfTestPairs; p++) {
		predTest[p] = V.row(tCols[p]).dot(U.row(tRows[p]));
	}
	predTest = predTest.array() + meanTrainValue;

	if (hasRowSideInfo) {
		rsMUs = rowSideMatrix * Us;
		for (int p = 0; p < numOfTestPairs; p++)
			predTest[p] += V.row(tCols[p]).dot(rsMUs.row(tRows[p]));
	}
}

double MF::testError() {
	if (verbosity > 2)
		std::cout << "testError()" << std::endl;
	predictTest();
	double fid = (predTest - tVals).norm();
	return sqrt(fid * fid / numOfTestPairs);
}

double MF::testErrorPar(int size, double* vec) {
	if(size != numOfTestPairs){
		std::cout << "testErrorPar(): size !=numOfTestPairs" << std::endl;
		exit(0);
	}
	if (verbosity > 2)
		std::cout << "testErrorPar()" << std::endl;
	Eigen::Map<Vector> tmp(vec, numOfTestPairs);
	double fid = (tmp - tVals).norm();
	return sqrt(fid * fid / numOfTestPairs);
}

void MF::PoldToP() {
	if (verbosity > 2)
		std::cout << "PoldToP()" << std::endl;
	U = U_old;
	V = V_old;
	if (hasRowSideInfo)
		Us = Us_old;
}

void MF::PtoPold() {
	if (verbosity > 2)
		std::cout << "PtoPold()" << std::endl;
	U_old = U;
	V_old = V;
	if (hasRowSideInfo)
		Us_old = Us;
}

void MF::PglobToP() {
	if (verbosity > 2)
		std::cout << "PglobToP()" << std::endl;
	U = U_glob;
	V = V_glob;
	if (hasRowSideInfo)
		Us = Us_glob;
}

void MF::PtoPglob() {
	if (verbosity > 2)
		std::cout << "PtoPglob()" << std::endl;
	U_glob = U;
	V_glob = V;
	if (hasRowSideInfo)
		Us_glob = Us;
}

void MF::eraseMomentums() {
	if (verbosity > 2)
		std::cout << "eraseMomentums()" << std::endl;
	U_m.fill(0.);
	V_m.fill(0.);
	if (hasRowSideInfo)
		Us_m.fill(0.);
}

void MF::saveFactors() {
	if (verbosity > 2)
		std::cout << "saveFactors()" << std::endl;
	std::ofstream file("U.txt");
	if (file.is_open())
		file << U << '\n';
	file.close();
	file.open("V.txt");
	if (file.is_open())
		file << V << '\n';
	file.close();
	if (hasRowSideInfo) {
		file.open("Us.txt");
		if (file.is_open())
			file << Us << '\n';
		file.close();
	}
}

DenseMatrix& MF::getPrediction() {
	P = Eigen::MatrixXd::Zero(numOfRows, numOfCols);
	P.noalias() = U*V.transpose();
	if (hasRowSideInfo)
		P.noalias() += (rowSideMatrix*Us)*V.transpose();
	P = P.array() + meanTrainValue;
	return P;
}

Vector& MF::getValPrediction(){
	predictVal();
	return predVal;
}

Vector& MF::getTestPrediction(){
	predictTest();
	return predTest;
}

bool MF::doesHaveRowSideInfo() {
	return hasRowSideInfo;
}
