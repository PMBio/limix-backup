/*
 * CLinearMean.h
 *
 *  Created on: Jan 3, 2012
 *      Author: clippert
 */

#ifndef CLINEARMEAN_H_
#define CLINEARMEAN_H_

#include "ADataTerm.h"

namespace gpmix {

class CLinearMean: public ADataTerm {
	MatrixXd weights;
	MatrixXd fixedEffects;
public:
	CLinearMean();
	CLinearMean(MatrixXd& Y, MatrixXd& weights, MatrixXd& fixedEffects);
	virtual ~CLinearMean();
	inline MatrixXd evaluate(){ return (this->getY() - (this->fixedEffects * this->weights)); };
	inline MatrixXd gradParams(){ return ( -fixedEffects ); };
	inline MatrixXd gradY(){return MatrixXd::Ones(Y.rows(),Y.cols());};
};

} /* namespace gpmix */
#endif /* CLINEARMEAN_H_ */
