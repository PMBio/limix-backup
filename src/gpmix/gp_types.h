/*
 * gp_types.h
 *
 *  Created on: Nov 10, 2011
 *      Author: stegle
 */

#ifndef GP_TYPES_H_
#define GP_TYPES_H_

#include <string>
using namespace std;
#include <gpmix/matrix/matrix_helper.h>


class CGPHyperParams {

protected:
	//possible hyperparameter object supported; we could also use a dictionary for that
	//but that maybe slower?
	//VectorXd covar;
	//VectorXd lik;
	//VectorXd mean;
	//MatrixXd x;

	VectorXd param_array;

public:

	CGPHyperParams();
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

	VectorXd get(string name)
	{
		return VectorXd(1,1);
	}

	VectorXs getNames()
	{
		return VectorXs(1,1);
	}

};



#endif /* GP_TYPES_H_ */
