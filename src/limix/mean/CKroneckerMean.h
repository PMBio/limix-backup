/*
 * CKroneckerSumMean.h
 *
 *  Created on: Jan 19, 2012
 *      Author: clippert
 */

#ifndef CKRONECKERSUMMEAN_H_
#define CKRONECKERSUMMEAN_H_

#include "ADataTerm.h"
#include "CLinearMean.h"
#include "limix/utils/matrix_helper.h"
namespace limix {

#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//%sptr(gpmix::CKroneckerSumMean)
#endif

class CKroneckerSumMean : public CLinearMean {
protected:
	MatrixXd A;
public:
	CKroneckerSumMean();
	CKroneckerSumMean(MatrixXd& Y, MatrixXd& fixedEffects, MatrixXd& A);
	CKroneckerSumMean(MatrixXd& Y, MatrixXd& weights, MatrixXd& fixedEffects, MatrixXd& A);
	virtual ~CKroneckerSumMean();

	virtual void setA(const MatrixXd& A);
	virtual void agetA(MatrixXd* out)
	{
		//generate the pseudo Design matrix that is equivalent ot the one used by CKroneckerSumMean
		//TODO: think about implementing CLinearMean as a special case of CKroneckerSumMean?!
		(*out) = A;
	}
	virtual void setWeightsOLS(const MatrixXd& Y);

	virtual void aEvaluate(MatrixXd* outY);
	virtual void aPredictY(MatrixXd* outY) const;
	virtual void aGradParams(MatrixXd* outGradParams, const MatrixXd* KinvY);
	inline virtual void setWeightsOLS(){setWeightsOLS(this->Y);};
	inline void checkDimensions(const MatrixXd& Y, const bool checkStrictWeights) const throw (CGPMixException);
	virtual inline std::string getName() const { return "CKoneckerFixedTerm"; };
	inline muint_t getDimFixedEffects() const { return this->fixedEffects.cols(); };
	virtual muint_t getColsParams()
	{
		return (muint_t) this->A.rows();
	}
};
typedef sptr<CKroneckerSumMean> PKroneckerMean;

inline void CKroneckerSumMean::checkDimensions(const MatrixXd& Y, const bool checkStrictWeights = true) const throw (CGPMixException)
{
	if (Y.rows() != this->fixedEffects.rows() && (muint_t)Y.cols() != this->getNTargets())
	{
		std::ostringstream os;
		os << this->getName() << ": Number of number samples and number of targets specified do not match with given Y. Y.rows() = " << Y.rows() << ", Y.cols() = " << Y.cols() << ", nSamples = " << fixedEffects.rows() << ", nTargets = " << this->getNTargets();
		throw CGPMixException(os.str());
	}
	//this->checkDimensions(this->fixedEffects, this->weights, this->A, true, checkStrictWeights, true);
}

} /* namespace limix */
#endif /* CKroneckerSumMean_H_ */
