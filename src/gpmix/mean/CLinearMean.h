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
	void zeroInitWeights();
public:
	CLinearMean();
	CLinearMean(MatrixXd& Y, MatrixXd& weights, MatrixXd& fixedEffects);
	CLinearMean(MatrixXd& Y, MatrixXd& fixedEffects);
	virtual ~CLinearMean();

	void aEvaluate(MatrixXd* outY);
	void aGradParams(MatrixXd* outGradParams);

	virtual void setParams(MatrixXd& weightMatrix);
	virtual void setfixedEffects(MatrixXd& fixedEfects);
	virtual void aGetParams(MatrixXd* outParams);
	virtual void aGetFixedEffects(MatrixXd* outFixedEffects);

	virtual inline MatrixXd getFixedEffects(){MatrixXd outFixedEffects; aGetFixedEffects(&outFixedEffects); return outFixedEffects;}
};

} /* namespace gpmix */
#endif /* CLINEARMEAN_H_ */
