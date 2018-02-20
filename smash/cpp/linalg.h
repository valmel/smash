#ifndef LINALG_H
#define LINALG_H

#define USE_OPENMP true

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <stdlib.h>

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SparseMatrix;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> DenseMatrix;
typedef Eigen::VectorXd Vector;

inline void ScipyCooToEigenSparse(const int nrows, const int ncols,
		const int ndata, const int * const rows, const int * const cols,
		const double * const values, SparseMatrix &M) {
	M.resize(nrows, ncols);
	typedef Eigen::Triplet<double> T;
	std::vector<T> tripletList;
	tripletList.reserve(ndata);
	for (int n = 0; n < ndata; n++) {
		tripletList.push_back(T(rows[n], cols[n], values[n]));
	}
	M.setFromTriplets(tripletList.begin(), tripletList.end());
}

inline void SparseMatrixToCoo(const SparseMatrix &M, int & nData, Eigen::VectorXi & rows,
		Eigen::VectorXi & cols, Eigen::VectorXd & values) {
	nData = M.nonZeros();
	rows.resize(nData);
	cols.resize(nData);
	values.resize(nData);
	int idx = 0;
	for (int r = 0; r < M.outerSize(); ++r)
	  for (SparseMatrix::InnerIterator it(M, r); it; ++it)
	  {
	    rows[idx] = r;
	    cols[idx] = it.col();
	    values[idx] = it.value();
	    idx++;
	  }
}

inline void SparseMatrixRowRangePer(const int bRow, const int eRow,
		const std::vector<int> & per, const SparseMatrix &M, int & nData,
		Eigen::VectorXi & rows, Eigen::VectorXi & cols,
		Eigen::VectorXd & values) {
	assert(bRow >= 0);
	assert(bRow < eRow);
	assert(eRow <= M.rows());
	int r, i, idx;

	//compute the number of data points
	nData = 0;
	for (r = bRow; r < eRow; ++r)
		nData = nData
				+ (M.outerIndexPtr()[per[r] + 1] - M.outerIndexPtr()[per[r]]);

	rows.resize(nData);
	cols.resize(nData);
	values.resize(nData);

	idx = 0;
	for (r = bRow; r < eRow; ++r) {
		for (i = M.outerIndexPtr()[per[r]]; i < M.outerIndexPtr()[per[r] + 1];
				++i) {
			rows[idx] = r;
			cols[idx] = M.innerIndexPtr()[i];
			values[idx] = M.valuePtr()[i];
			idx++;
		}
	}
}

inline int getMaxNumberOfRowNonzeros(const SparseMatrix &M) {

	int max = 0;
	for (int r = 0; r < M.rows(); ++r)
		if ((M.outerIndexPtr()[r + 1] - M.outerIndexPtr()[r]) > max)
			max = M.outerIndexPtr()[r + 1] - M.outerIndexPtr()[r];

	return max;
}

#endif // LINALG_H
