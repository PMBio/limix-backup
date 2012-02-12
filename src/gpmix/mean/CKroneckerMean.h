/*
 * CKroneckerMean.h
 *
 *  Created on: Jan 19, 2012
 *      Author: clippert
 */

#ifndef CKRONECKERMEAN_H_
#define CKRONECKERMEAN_H_

#include "ADataTerm.h"
#include "CLinearMean.h"
#include "gpmix/utils/matrix_helper.h"
namespace gpmix {

#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//%shared_ptr(gpmix::CKroneckerMean)
#endif

class CKroneckerMean : public gpmix::CLinearMean {
	MatrixXd A;
public:
	CKroneckerMean(muint_t nSamples, muint_t nTargets);
	CKroneckerMean(MatrixXd& Y, MatrixXd& weights, MatrixXd& fixedEffects, MatrixXd& A);
	virtual ~CKroneckerMean();

	virtual void setA(MatrixXd& A);
	virtual void setWeightsOLS(const MatrixXd& Y);

	virtual void aEvaluate(MatrixXd* outY);
	virtual void aPredictY(MatrixXd* outY) const;
	virtual void aGradParams(MatrixXd* outGradParams, const MatrixXd* KinvY);
	inline virtual void setWeightsOLS(){setWeightsOLS(this->Y);};
	inline void checkDimensions(const MatrixXd& fixedEffects, const MatrixXd& weights, const MatrixXd& A, const bool checkStrictFixedEffects = false, const bool checkStrictWeights = false, const bool checkStrictA = false) const throw (CGPMixException);
	inline void checkDimensions(const MatrixXd& Y, const bool checkStrictWeights) const throw (CGPMixException);
	virtual inline std::string getName() const { return "CKoneckerFixedTerm"; };
	inline muint_t getDimFixedEffects() const { return this->fixedEffects.cols(); };
	virtual muint_t getColsParams() {
		return (muint_t) this->A.rows();
	}

	inline void makeSync() const {}
};


inline void CKroneckerMean::checkDimensions(const MatrixXd& Y, const bool checkStrictWeights = true) const throw (CGPMixException)
{
	if (Y.rows() != this->fixedEffects.rows() && (muint_t)Y.cols() != this->getNTargets())
	{
		std::ostringstream os;
		os << this->getName() << ": Number of number samples and number of targets specified do not match with given Y. Y.rows() = " << Y.rows() << ", Y.cols() = " << Y.cols() << ", nSamples = " << fixedEffects.rows() << ", nTargets = " << this->getNTargets();
		throw gpmix::CGPMixException(os.str());
	}
	this->checkDimensions(this->fixedEffects, this->weights, this->A, true, checkStrictWeights, true);
}

inline void CKroneckerMean::checkDimensions(const MatrixXd& fixedEffects, const MatrixXd& weights, const MatrixXd& A, const bool checkStrictFixedEffects, const bool checkStrictWeights, const bool checkStrictA) const throw (CGPMixException)
{
#if 0
	bool notIsnullFixed = false;
	bool notIsnullweights = false;
	bool notIsnullA = false;

	//if no strict testing is performed, we test, if the argument has been set
	if(!checkStrictFixedEffects)
	{
		notIsnullFixed = !isnull(fixedEffects);
	}
	if(!checkStrictWeights)
	{
		notIsnullweights = !isnull(weights);
	}
	if(!checkStrictA)
	{
		notIsnullA = !isnull(A);
	}

	if (notIsnullA && ((muint_t)A.cols() != this->getNTargets()) )
	{
		ostringstream os;
		os << this->getName() << ": Number of cols in A and number targets specified do not match: A = " << A.cols() << ", number targets = " << this->getNTargets();
		throw gpmix::CGPMixException(os.str());
	}

	if (notIsnullweights && notIsnullFixed && (weights.rows()) != fixedEffects.cols() ){
		ostringstream os;
		os << this->getName() << ": Number of weights and fixed effects do not match. number fixed effects = " << fixedEffects.cols() << ", number weights = " << weights.rows();
		throw gpmix::CGPMixException(os.str());
	}
	if ( notIsnullweights && notIsnullA && (weights.cols()) != A.rows() ){
			ostringstream os;
			os << this->getName() << ": #Cols of weights and #Rows of A do not match: weights = " << weights.cols() << ", A = " << A.rows();
			throw gpmix::CGPMixException(os.str());
		}
#endif
}

} /* namespace gpmix */
#endif /* CKRONECKERMEAN_H_ */
