/*
 * gp_types.h
 *
 *  Created on: Nov 10, 2011
 *      Author: stegle
 */

#ifndef GP_TYPES_H_
#define GP_TYPES_H_

#include <gpmix/matrix/matrix_helper.h>


class CGPHyperParams {

protected:
	//possible hyperparameter object supported; we could also use a dictionary for that
	//but that maybe slower?
	VectorXd covar;
	VectorXd lik;
	VectorXd mean;
	MatrixXd x;

public:

	//constructors
	//default
	//TODO:
/*
	CGPHyperParams();
	//from a list of params
	CGPHyperParams(VectorXd param_list);
	VectorXd toParamArray()
	{
		return VectorXd(1,1);
	}
*/

	//TODO: think about copying or not copying?
	void set_covar(VectorXd covar)
	{
		this->covar = covar;
	}
	void set_lik(VectorXd lik)
	{
		this->lik = lik;
	}
	void set_mean(VectorXd mean)
	{
		this->mean = mean;
	}
	void set_x(MatrixXd x)
	{
		this->x = x;
	}

	VectorXd get_covar()
	{
		return covar;
	}
	VectorXd get_lik()
	{
		return lik;
	}
	VectorXd get_mean()
	{
		return mean;
	}
	MatrixXd get_x()
	{
		return x;
	}
};



#endif /* GP_TYPES_H_ */
