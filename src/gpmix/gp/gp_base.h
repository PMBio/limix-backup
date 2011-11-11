/*
 * gp_base.h
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */

#ifndef GP_BASE_H_
#define GP_BASE_H_

#include <gpmix/covar/covariance.h>
#include <gpmix/likelihood/likelihood.h>
#include <string>
#include <map>
#include <gpmix/types.h>
using namespace std;

namespace gpmix {

class CGPHyperParams {

protected:

	VectorXd param_array;
	map<string,VectorXd> param_map;

public:

	CGPHyperParams()
	{
		//empty constructur
	}
	//from a list of params

	VectorXd getParamArray()
	{
		return param_array;
	}

	void setParamArray(VectorXd param)
	{
		//TODO: check that length is ok
		this->param_array = param;
	}

	void set(string name,VectorXd value)
	{
		return;
	}

	VectorXd get(const string& name)
	{
		return param_map[name];
	}

	VectorXs getNames()
	{
		return VectorXs(1,1);
	}

};



class CGPbase {
protected:
	ACovarianceFunction& covar;
	ALikelihood& lik;

//	virtual double _LML_covar(CGPHyperParams& parmas);
//	virtual VectorXd _LMLgrad_covar(CGPHyperParams& params);

public:
	CGPbase(ACovarianceFunction& covar, ALikelihood& lik);
	virtual ~CGPbase();

//TODO: add interface that is suitable for optimizer
// virtual double LML(double* params);
// virtual void LML(double* params, double* gradients);


	virtual float_t LML(CGPHyperParams& hyperparams);
	virtual CGPHyperParams LMLgrad(CGPHyperParams& hyperparams);

};

} /* namespace gpmix */
#endif /* GP_BASE_H_ */
