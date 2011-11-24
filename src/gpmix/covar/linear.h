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
	MatrixXd K_grad_X(const muint_t d) const;
	MatrixXd K_grad_param(const muint_t i) const;

	//gradient of K(Xstar,X)
	virtual MatrixXd Kcross_grad_X(const CovarInput& Xstar, const muint_t d) const;
	virtual MatrixXd Kdiag_grad_X(const muint_t d) const;

	//class information
	inline string getName() const {return "CCovLinearISO";};

};

#ifndef SWIG
class CCovLinearARD: public ACovarianceFunction  {
public:
	CCovLinearARD(const muint_t dimensions) : ACovarianceFunction(dimensions)
	{
	}

	~CCovLinearARD();

	//compute K(Xstar,X)
	MatrixXd Kcross( const CovarInput& Xstar ) const;
	VectorXd Kdiag() const;
	MatrixXd K_grad_x(const muint_t d) const;
	MatrixXd K_grad_param(const muint_t i) const;

	//gradient of K(Xstar,X)
	MatrixXd Kcross_grad_X(const CovarInput& Xstar, const muint_t d) const;
	MatrixXd Kdiag_grad_X(const muint_t d) const;

	//class information
	inline string getName() const{ return "CovLinearARD";}
};
#endif

} /* namespace gpmix */
#endif /* LINEAR_H_ */
