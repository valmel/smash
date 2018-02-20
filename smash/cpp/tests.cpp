#include <chrono>
#include <thread>
#include "mf.h"

int main(int argc, const char * argv[]){
#ifdef _OPENMP
	std::cout << "Have OMP" << _OPENMP << "\n";
#endif
#ifdef EIGEN_HAS_OPENMP
	std::cout << "Have EIGEN_HAS_OPENMP\n";
#endif
#ifdef _OPENMP
	std::cout << "Eigen::nbThreads() = " << Eigen::nbThreads() << "\n";
#endif
	Eigen::initParallel();
	//omp_set_num_threads(4);
	MF mf = MF();
	mf.setVerbosity(1);
	mf.loadTrainMatrix("../../data/chembl-IC50-346targets_train.mtx");
	mf.loadValMatrix("../../data/chembl-IC50-346targets_val.mtx");
	mf.loadTestMatrix("../../data/chembl-IC50-346targets_test.mtx");
	mf.loadRowSideMatrix("../../data/chembl-IC50-compound-feat.mm");
	mf.setNumOfBatches(9);
	mf.setInitialLR(3200.);
	mf.setInitialAlpha(0.32);
	mf.setLatentDim(100.);
	mf.setInitializationScalingFactor(0.3);
	mf.setMaxEpoch(400);
	mf.train();
	std::this_thread::sleep_for(std::chrono::milliseconds(1000));
}
