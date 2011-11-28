/*
 * CCovSqexpARD.h
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */

#ifndef CCOVSQEXPARD_H_
#define CCOVSQEXPARD_H_

#include "covariance.h"

namespace gpmix {

class CCovSqexpARD: public gpmix::ACovarianceFunction {
public:
	CCovSqexpARD(): ACovarianceFunction(1)
	{
	}

	~CCovSqexpARD();

	//overloaded pure virtual functions:
	virtual void Kcross(MatrixXd* out, const CovarInput& Xstar ) const;
	virtual void Kgrad_param(MatrixXd* out,const muint_t i) const;
	virtual void Kcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const;
	virtual void Kdiag_grad_X(MatrixXd* out,const muint_t d) const;

	//class information
	inline string getName() const{ return "CovLinearARD";}

	//redefine setX
	inline virtual void setX(const CovarInput& X);

}; //end class CCovSqexpARD


inline void CCovSqexpARD::setX(const CovarInput & X)
{
	this->X = X;
	this->insync = false;
	this->numberParams = 1+X.cols();
}

} /* namespace gpmix */
#endif /* CCOVSQEXPARD_H_ */
