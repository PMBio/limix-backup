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
	muint_t numMaxCovariances; //!< internal varaince, controlling the maximum number of cavariances in a combinators, this may depend on the CF (default -1) - no limi
public:
	/*!
	 * Constructor for abstract type of MultiPleCovariances
	 * \param covariances: pre-initialized vector with sub covarinaces
	 * \param numMaxCovariances: maximum number of covariances to permit
	 */
	AMultiCF(const ACovarVec& covariances,muint_t numMaxCovariances=999);

	/*!
	 * Constructor for abstract type of MultiPleCovariances
	 * \param numCovariancesInit: number of covarinaces to expect during init
	 * \param numMaxCovariances: maximum number of covariances to permit
	 */
	AMultiCF(muint_t numCovariancesInit=0,muint_t numMaxCovariances=999);

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
%rename(getParams) CLinCombCF::agetParams;
#endif
/*
* Kronecker function for pairs of covariances
*/
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
	virtual void aK(MatrixXd* out) const throw (CGPMixException);
	virtual void aKdiag(VectorXd* out) const throw (CGPMixException);
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
    
#if (!defined(SWIG_FILE_WITH_INIT) && defined(SWIG))
//%sptr(gpmix::CKroneckerCF)
#endif
    
/*!
 * \brief Kronecker structure combinator for two covariances
 *
 * Combines two covarianes Kc \kron Kr. The class supports both propper Kronecker structures
 * as well as "soft Kronecker, where
 */
class CKroneckerCF: public AMultiCF
{
protected:
	//optional indicator vector to pull together the kronecker structure
	MatrixXi kroneckerIndicator; //!< indicator with kronecker structure N row (total samples), with indexes for [row,col] to create sof Kronecker structures
public:
	CKroneckerCF();
	CKroneckerCF(PCovarianceFunction col,PCovarianceFunction row);
	virtual ~CKroneckerCF();
	virtual muint_t Kdim() const throw(CGPMixException);

	//Access to vecCovariances
	virtual void setRowCovariance(PCovarianceFunction cov);
	virtual void setColCovariance(PCovarianceFunction cov);
	PCovarianceFunction getRowCovariance() throw (CGPMixException);
	PCovarianceFunction getColCovariance() throw (CGPMixException);

	/*!
	 * set KronecekerIndicator, which needs to be N x 2 with indicse for row & column of individual elements
	 */
	void setKroneckerIndicator(const MatrixXi& kroneckerIndicator);

	/*!
	 * getKroneckerIndicator.
	 */
	void getKroneckerIndicator(MatrixXi* out) const;


	/*!
	 * is a kronecker covariance? if a KroneckerIndicator is set this object is not a propper Kronecker.
	 */
	bool isKronecker() const;

	//X handling
	virtual void setX(const CovarInput& X) throw (CGPMixException) {};
	virtual void agetX(CovarInput* Xout) const throw (CGPMixException) {};
	virtual void setXcol(const CovarInput& X,muint_t col) throw (CGPMixException) {};
	virtual void setXr(const CovarInput& Xr) throw (CGPMixException);
	virtual void setXc(const CovarInput& Xc) throw (CGPMixException);

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

	/*!
	 * Convenience functin to generator a a Kronecker Indicator matrix ([N*P,2])
	 * which is fully kroneckerized. This function is used to perform missing value imputations, etc.
	 * \see CVarianceDecomposition
	 * \param Ncols: number of samples (row covariance)
	 * \param Nrow: number of individuals (col covariance)
	 */
	static void createKroneckerIndex (MatrixXi* out,muint_t Ncols, muint_t Nrows);
};
typedef sptr<CKroneckerCF> PKroneckerCF;

} //end namespace limix




#endif /* COMBINATORS_H_ */
