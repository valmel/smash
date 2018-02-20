#ifndef SASGD_H
#define SASGD_H

#include "sgd.h"
//#include <limits>

class SASGD: public SGD {
protected:
	double alphaD = 1.5;
	double LRD = 1.0;
private:
	int updatedEpoch;
	double prevRMSEafterUpdate;
	double RMSEafterUpdate;
	//members
public:
	SASGD();
	virtual ~SASGD();
	void setAlphaDecay(double d = 1.5) {
		alphaD = d;
	}
	void setLRDecay(double d = 1.0) {
		LRD = d;
	}
	virtual void updateParameters(int epoch);
};

#endif  // SASGD_H
