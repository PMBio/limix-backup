/*
 * combinators.h
 *
 *  Created on: Dec 28, 2011
 *      Author: stegle
 */

#ifndef COMBINATORS_H_
#define COMBINATORS_H_

#include <vector>
#include "limix/covar/covariance.h"


namespace limix {

//Define a vector of covariances which is needed to represent the sum and product CF
typedef std::vector<PCovarianceFunction> ACovarVec;
//SWIG template declaration:
//TODO: swig vector
#if (!defined(SWIG_FILE_WITH_INIT) && defined(SWIG))
//%template(ACovarVec) std::vector<gpmix::PCovarianceFunction>;
//%sptr(gpmix::AMultiCF)
#endif

class AMultiCF : public ACovarianceFunction
{
protected:
	ACovarVec vecCovariances;
public:
	AMultiCF(const ACovarVec& covariances);
	AMultiCF(const muint_t numCovariances=0);
	//destructors
	virtual ~AMultiCF();
	virtual muint_t Kdim() const throw(CGPMixException);

	//sync stuff
	void addSyncChild(Pbool l);
	void delSyncChild(Pbool l);

	//access to covariance arrays
	void addCovariance(PCovarianceFunction covar) throw (CGPMixException);
	void setCovariance(muint_t i,PCovarianceFunction covar) throw (CGPMixException);
	PCovarianceFunction getCovariance(muint_t i) throw (CGPMixException);

	virtual muint_t getNumberDimensions() const throw (CGPMixException);
	virtual void setNumberDimensions(muint_t numberDimensions) throw (CGPMixException);
	virtual muint_t getNumberParams() const;

	//setX and getX
	virtual void setX(const CovarInput& X) throw (CGPMixException);
	virtual void agetX(CovarInput* Xout) const throw (CGPMixException);
	virtual void setXcol(const CovarInput& X,muint_t col) throw (CGPMixException);

	//set and get Params
	virtual void setParams(const CovarParams& params);
	virtual void agetParams(CovarParams* out) const;
	//set and get Param masks
	virtual void agetParamMask(CovarParams* out) const;
	virtual void setParamMask(const CovarParams& params);
	//get parameter bounds
	virtual void agetParamBounds0(CovarParams* lower,CovarParams* upper) const;
    virtual void agetParamBounds(CovarParams* lower,CovarParams* upper) const;
	virtual void setParamBounds(const CovarParams& lower, const CovarParams& upper)  throw (CGPMixException);
};

#if (!defined(SWIG_FILE_WITH_INIT) && defined(SWIG))
//%sptr(gpmix::CSumCF)
#endif

class CSumCF : public AMultiCF {
public:
	CSumCF(const ACovarVec& covariances);
	CSumCF(const muint_t numCovariances=0);
	//destructors
	virtual ~CSumCF();

	//overloaded pure virtual members
	virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException);
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException);
	virtual void aKgrad_param(MatrixXd* out,const muint_t i) const throw(CGPMixException);
    virtual void aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const throw(CGPMixException);
	virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const throw(CGPMixException);
	virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const throw(CGPMixException);
	//optional overloadings:
	virtual void aK(MatrixXd* out) const throw (CGPMixException);
	virtual void aKdiag(VectorXd* out) const throw (CGPMixException);
	virtual void aKgrad_X(MatrixXd* out,const muint_t d) const throw(CGPMixException);
	virtual std::string getName() const;
};
typedef sptr<CSumCF> PSumCF;

#if (!defined(SWIG_FILE_WITH_INIT) && defined(SWIG))
%rename(getParams) ACovarianceFunction::agetParams;
#endif

class CLinCombCF : public AMultiCF {
protected:
	VectorXd coeff;
public:
	CLinCombCF(const ACovarVec& covariances);
	CLinCombCF(const muint_t numCovariances=0);
	//destructors
	virtual ~CLinCombCF();
	//linear coefficients
	virtual void setCoeff(const VectorXd& coeff);
	virtual void agetCoeff(VectorXd* out) const;


	//overloaded pure virtual members
	virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException);
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException);
	virtual void aKgrad_param(MatrixXd* out,const muint_t i) const throw(CGPMixException);
    virtual void aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const throw(CGPMixException);
	virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const throw(CGPMixException);
	virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const throw(CGPMixException);
	//optional overloadings:
	virtual void aK(MatrixXd* out) const;
	virtual void aKdiag(VectorXd* out) const;
	virtual void aKgrad_X(MatrixXd* out,const muint_t d) const throw(CGPMixException);
	virtual std::string getName() const;
};
typedef sptr<CLinCombCF> PLinCombCF;




#if (!defined(SWIG_FILE_WITH_INIT) && defined(SWIG))
//%sptr(gpmix::CProductCF)
#endif

class CProductCF : public AMultiCF {
public:
	CProductCF(const ACovarVec& covariances);
	CProductCF(const muint_t numCovariances=0);
	//destructors
	virtual ~CProductCF();

	//overloaded pure virtual members
	virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException);
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException);
	virtual void aKgrad_param(MatrixXd* out,const muint_t i) const throw(CGPMixException);
    virtual void aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const throw(CGPMixException);
	virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const throw(CGPMixException);
	virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const throw(CGPMixException);
	//optional overloadings:
	virtual void aK(MatrixXd* out) const throw (CGPMixException);
	virtual void aKdiag(VectorXd* out) const throw (CGPMixException);
	virtual void aKgrad_X(MatrixXd* out,const muint_t d) const throw(CGPMixException);
	virtual std::string getName() const;
};
typedef sptr<CProductCF> PProductCF;


/*
 * Kronecker function for pairs of covariances
 */
class CKroneckerCF: public AMultiCF
{
protected:
	//covariance fucntions for row and columns
	PCovarianceFunction rowCovar,colCovar;
	//optional indicator vector to pull together the kronecker structure
	MatrixXi kroneckerIndicator;

public:
	CKroneckerCF();
	CKroneckerCF(PCovarianceFunction row,PCovarianceFunction col);
	virtual ~CKroneckerCF();

	virtual void setRowCovariance(PCovarianceFunction cov);
	virtual void setColCovariance(PCovarianceFunction cov);

	//overloaded pure virtual members
	virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException);
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException);
	virtual void aKgrad_param(MatrixXd* out,const muint_t i) const throw(CGPMixException);
    virtual void aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const throw(CGPMixException);
	virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const throw(CGPMixException);
	virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const throw(CGPMixException);
	//optional overloaded functions for efficiency:
	virtual void aK(MatrixXd* out) const throw (CGPMixException);
	virtual void aKdiag(VectorXd* out) const throw (CGPMixException);
	virtual void aKgrad_X(MatrixXd* out,const muint_t d) const throw(CGPMixException);

	virtual std::string getName() const;
};


} //end namespace limix




#endif /* COMBINATORS_H_ */
