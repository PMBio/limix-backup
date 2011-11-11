/*
 * gp_base.h
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */

#ifndef GP_BASE_H_
#define GP_BASE_H_

#include <gpmix/matrix/matrix_helper.h>
#include <gpmix/covar/covariance.h>
#include <gpmix/gp_types.h>

#include <string>
using namespace std;

namespace gpmix {

class CGPbase {
protected:
	ACovarianceFunction& covar;

//	virtual double _LML_covar(CGPHyperParams& parmas);
//	virtual VectorXd _LMLgrad_covar(CGPHyperParams& params);

public:
	CGPbase(ACovarianceFunction& covar);
	virtual ~CGPbase();

//TODO: add interface that is suitable for optimizer
// virtual double LML(double* params);
// virtual void LML(double* params, *double gradients);


//	virtual double LML(CGPHyperParams& hyperparams);
//	virtual CGPHyperParams LMLgrad(CGPHyperParams& hyperparams);

};

} /* namespace gpmix */
#endif /* GP_BASE_H_ */
