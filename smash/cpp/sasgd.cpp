#include "sasgd.h"

SASGD::SASGD() :
		SGD() {
	//SGD::SGD();
	setAlphaDecay();
	setLRDecay();
}

SASGD::~SASGD() {
}

void SASGD::updateParameters(int epoch) {
	if (epoch == 0) {
		updatedEpoch = -2;
		prevRMSEafterUpdate = std::numeric_limits<double>::max();
		RMSEafterUpdate = prevRMSEafterUpdate - 1.;
	}

	if (epoch == updatedEpoch + 1) {
		prevRMSEafterUpdate = RMSEafterUpdate;
		RMSEafterUpdate = err_valid[epoch];
	}

	if (RMSEafterUpdate > prevRMSEafterUpdate) {
		globToLatent();
		breakSignal = true;
	}

	if (epoch > minNumOfEpochs and err_valid[epoch] > err_valid[epoch - 1]) {
		alpha = std::max(alpha / alphaD, 0.01);
		LR = LR / (alphaD * LRD);
		latentToOld();
		latentToGlob();
		eraseMomentum();
		updatedEpoch = epoch;

		if (verbosity > 0) {
			std::cout << " alpha = " << alpha << " LR = " << LR << std::endl;
		}
	}
}
