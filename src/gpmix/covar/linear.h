/*
 * Linear.h
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */

#ifndef LINEAR_H_
#define LINEAR_H_

#include <gpmix/covar/covariance.h>

namespace gpmix {

class CCovLinearISO: public ACovarianceFunction  {
public:
	CCovLinearISO() : ACovarianceFunction(1)
	{
	}

	~CCovLinearISO();

	//compute K(Xstar,X)
	MatrixXd Kcross( const CovarInput& Xstar ) const;
	VectorXd Kdiag() const;
	MatrixXd K_grad_X(const uint_t d) const;
	MatrixXd K_grad_param(const uint_t i) const;

	//gradient of K(Xstar,X)
	virtual MatrixXd Kcross_grad_X(const CovarInput& Xstar, const uint_t d) const;
	virtual MatrixXd Kdiag_grad_X(const uint_t d) const;

	//class information
	inline string getName() const {return "CCovLinearISO";};

};


class CCovLinearARD: public ACovarianceFunction  {
public:
	CCovLinearARD(const uint_t dimensions) : ACovarianceFunction(dimensions)
	{
	}

	~CCovLinearARD();

	//compute K(Xstar,X)
	MatrixXd Kcross( const CovarInput& Xstar ) const;
	VectorXd Kdiag() const;
	MatrixXd K_grad_x(const uint_t d) const;
	MatrixXd K_grad_param(const uint_t i) const;

	//gradient of K(Xstar,X)
	MatrixXd Kcross_grad_X(const CovarInput& Xstar, const uint_t d) const;
	MatrixXd Kdiag_grad_X(const uint_t d) const;

	//class information
	inline string getName() const{ return "CovLinearARD";}
};


} /* namespace gpmix */
#endif /* LINEAR_H_ */
