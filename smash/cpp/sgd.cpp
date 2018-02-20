#include "sgd.h"

//using namespace std;

SGD::SGD() {
	initializeMemory();
}

SGD::~SGD() {
}

static volatile bool sigintRecieved = false;

void sigintHandler(int dummy) {
  sigintRecieved = true;
  std::cout << "[Received Ctrl-C. Stopping after finishing the current batch.]" << std::endl;
}

void SGD::initializeMemory() {
	err_train.resize(maxNumOfEpoch);
	err_valid.resize(maxNumOfEpoch);
	err_test.resize(maxNumOfEpoch);
}

void SGD::trainOneBatch(int batch) {
	prepareBatch(batch);
	computeRegTerms();
	computePredictions();
	computeStochGrads();
	updateMomentums();
	oneGradStep();
}

void SGD::updateParameters(int e) {
	if (e > minNumOfEpochs
			&& (err_valid[e - 1] - err_valid[e]) / err_valid[e - 1] < epsStop) {
		oldToLatent();
		breakSignal = true;
	}
}

void SGD::train() {
	initializeTraining();
	signal(SIGINT, sigintHandler);
	breakSignal = false;
	auto t_start = std::chrono::high_resolution_clock::now();
	for (int e = 0; e < maxNumOfEpoch; e++) {
		epoch = e;
		permuteData();
		// preserve the current latent parameters
		latentToOld();
		// go through all batches
		for (int b = 0; b < numOfBatches; b++) {
			auto b_start = std::chrono::high_resolution_clock::now();
			trainOneBatch(b);
			if (verbosity > 1) {
				auto t_now = std::chrono::high_resolution_clock::now();
				std::cout << "Batch: " << b << ": took "
						<< std::chrono::duration<double, std::milli>(
								t_now - b_start).count() / 1000. << "s"
						<< std::endl;
			}
			if (verbosity > 2) {
				std::cout << "\t Training RMSE " << trainError()
						<< "; Validation RMSE " << validError()
						<< "; Test RMSE " << testError() << std::endl;
			}
			if (sigintRecieved == true) {
				sigintRecieved = false;
				breakSignal = true;
				break;
			}
		}
		// save the progress
		err_train[e] = trainError();
		err_valid[e] = validError();
		err_test[e] = testError();
		// printout
		if (verbosity > 0) {
			std::cout << "Epoch: " << e << " Training RMSE " << err_train[epoch]
					<< "; Validation RMSE " << err_valid[epoch]
					<< "; Test RMSE " << err_test[epoch] << std::endl;
		}

		updateParameters(e);

		if (breakSignal)
			break;

	}
	auto t_now = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration<double, std::milli>(t_now - t_start).count() / 1000.;
}
