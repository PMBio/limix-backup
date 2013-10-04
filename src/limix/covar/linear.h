/*
 * Linear.h
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */

#ifndef LINEAR_H_
#define LINEAR_H_

#include "limix/covar/covariance.h"

namespace limix {

#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//%sptr(gpmix::CCovLinearISO)
#endif
class CCovLinearISO: public ACovarianceFunction  {
public:
	CCovLinearISO(muint_t numberDimensions=1) : ACovarianceFunction(1)
	{
		this->setNumberDimensions(numberDimensions);
		initParams();
	}

	~CCovLinearISO();

	//overloaded pure virtual functions:
	void aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException);
	void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException);

	void aKgrad_param(MatrixXd* out,const muint_t i) const throw(CGPMixException);
    void aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const throw(CGPMixException);
	void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const throw(CGPMixException);
	void aKdiag_grad_X(VectorXd* out,const muint_t d) const throw(CGPMixException);

	//class information
	inline std::string getName() const {return "CCovLinearISO";};

};
typedef sptr<CCovLinearISO> PCovLinearISO;

#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//%sptr(gpmix::CCovLinearARD)
#endif
class CCovLinearARD: public ACovarianceFunction  {
public:
	CCovLinearARD(muint_t numberDimensions=1) : ACovarianceFunction(1)
	{
		this->setNumberDimensions(numberDimensions);
		initParams();
	}

	~CCovLinearARD();
	//overloaded virtuals
	virtual void setNumberDimensions(muint_t numberDimensions);

	//overloaded pure virtual functions:
	void aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException);
	void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException);
	void aKgrad_param(MatrixXd* out,const muint_t i) const throw(CGPMixException);
    void aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const throw(CGPMixException);
	void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const throw(CGPMixException);
	void aKdiag_grad_X(VectorXd* out,const muint_t d) const throw(CGPMixException);

	//class information
	inline std::string getName() const{ return "CovLinearARD";}
};
typedef sptr<CCovLinearARD> PCovLinearARD;



/* Delta kernel */
class CCovLinearISODelta: public ACovarianceFunction  {
public:
	CCovLinearISODelta(muint_t numberDimensions=1) : ACovarianceFunction(1)
	{
		this->setNumberDimensions(numberDimensions);
		initParams();
	}

	~CCovLinearISODelta();

	//overloaded pure virtual functions:
	void aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException);
	void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException);

	void aKgrad_param(MatrixXd* out,const muint_t i) const throw(CGPMixException);
    void aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const throw(CGPMixException);


	//class information
	inline std::string getName() const {return "CCovLinearISODelta";};

};
typedef sptr<CCovLinearISODelta> PCovLinearISODelta;




} /* namespace limix */
#endif /* LINEAR_H_ */
