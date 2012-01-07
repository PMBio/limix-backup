/*
 * CLinearMean.cpp
 *
 *  Created on: Jan 3, 2012
 *      Author: clippert
 */

#include "CLinearMean.h"

namespace gpmix {

CLinearMean::CLinearMean() : ADataTerm::ADataTerm() {
	// TODO Auto-generated constructor stub

}

CLinearMean::CLinearMean(MatrixXd& Y, MatrixXd& weights, MatrixXd& fixedEffects) : ADataTerm::ADataTerm(Y) {
	this->fixedEffects = fixedEffects;
	this->weights = weights;
}


CLinearMean::~CLinearMean() {
	// TODO Auto-generated destructor stub
}

} /* namespace gpmix */
