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

	void aEvaluate(MatrixXd* outY);
	void aGradParams(MatrixXd* outGradParams);

	inline MatrixXd evaluate(){ MatrixXd Y = MatrixXd(); aEvaluate(&Y); return Y; };
	inline MatrixXd gradParams(){ MatrixXd gradParams = MatrixXd(); aEvaluate(&gradParams); return gradParams;  };
};

} /* namespace gpmix */
#endif /* CLINEARMEAN_H_ */
