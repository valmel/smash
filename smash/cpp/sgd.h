#ifndef SGD_H
#define SGD_H

#include <vector>
#include <iostream>
//#include <iomanip>
#include <chrono>
#include <signal.h>

class SGD {
protected:
	int verbosity = 1;
	int epoch = 0;
	// pars
	double alpha = 0.2; // regularization
	double LR = 50.; // learning rate
	int latentDim = 10; // dimension of the latent space
	double isf = 1.; // scaling factor of the initialization
	double momentum = 0.8;
	int maxNumOfEpoch = 1000;
	int numOfBatches = 9;
	double epsStop = 1.e-6;
	int minNumOfEpochs = 3;
	// logs
	std::vector<double> err_train = std::vector<double>();
	std::vector<double> err_valid = std::vector<double>();
	std::vector<double> err_test = std::vector<double>();
	// termination logic
	bool breakSignal = false;
	//timing
	double duration = -1.;

public:
	SGD();
	virtual ~SGD();
	// parameter interface functions
	void setVerbosity(int iver = 1) {
		verbosity = iver;
	}
	void setInitialAlpha(double d = 0.2) {
		alpha = d;
	}
	void setInitialLR(double d = 50.) {
		LR = d;
	}
	void setLatentDim(int i = 10) {
		latentDim = i;
	}
	void setInitializationScalingFactor(double d = 1.) {
		isf = d;
	}
	void setMomentum(double d = 0.8) {
		momentum = d;
	}
	void setMaxEpoch(int i = 1000) {
		maxNumOfEpoch = i;
		initializeMemory();
	}
	void setNumOfBatches(int i = 9) {
		numOfBatches = i;
	}
	void setEpsStop(double d = 1.e-6) {
		epsStop = d;
	}
	void setMinimalNumOfEpochs(int i = 3) {
		minNumOfEpochs = i;
	}
	int getEpoch() {
		return epoch;
	}
	double getAlpha() {
		return alpha;
	}
	double getLR() {
		return LR;
	}
	double getDuration() {
		return duration;
	}
	//the main work happens here
	void train();

protected:
	// purely virtual functions
	virtual void initializeTraining() {};
	virtual void prepareBatch(int batch) {};
	virtual void computeRegTerms() {};
	virtual void computePredictions() {};
	virtual void computeLikelihoodGrads() {};
	virtual void computeRegGrads() {};
	virtual void aggregateGrads() {};
	virtual void computeStochGrads() {};
	virtual void updateMomentums() {};
	virtual void oneGradStep() {};
	virtual void permuteData() {};
	virtual double trainError() { return 0.; };
	virtual double validError() { return 0.; };
	virtual double testError() { return 0.; };
	virtual void oldToLatent() {};
	virtual void latentToOld() {};
	virtual void latentToGlob() {};
	virtual void globToLatent() {};
	virtual void eraseMomentum() {};
	void trainOneBatch(int batch);
	virtual void updateParameters(int epoch);

protected:
	void setDefaultAlgorithmPars();
	void initializeMemory();

};

#endif  // SGD_H
