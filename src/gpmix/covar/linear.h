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

	//overloaded pure virtual functions:
	void Kcross(MatrixXd* out, const CovarInput& Xstar ) const;
	void Kgrad_param(MatrixXd* out,const muint_t i) const;
	void Kcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const;
	void Kdiag_grad_X(VectorXd* out,const muint_t d) const;

	//class information
	inline string getName() const {return "CCovLinearISO";};

};

class CCovLinearARD: public ACovarianceFunction  {
public:
	CCovLinearARD() : ACovarianceFunction(1)
	{
	}

	~CCovLinearARD();
	//overloaded pure virtual functions:
	void Kcross(MatrixXd* out, const CovarInput& Xstar ) const;
	void Kgrad_param(MatrixXd* out,const muint_t i) const;
	void Kcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const;
	void Kdiag_grad_X(VectorXd* out,const muint_t d) const;

	//class information
	inline string getName() const{ return "CovLinearARD";}

	//redefine setX
	inline virtual void setX(const CovarInput& X);
};

inline void CCovLinearARD::setX(const CovarInput & X)
{
	this->X = X;
	this->insync = false;
	this->numberParams = 1+X.cols();
}



} /* namespace gpmix */
#endif /* LINEAR_H_ */
