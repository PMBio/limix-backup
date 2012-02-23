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
protected:
	MatrixXd A;
public:
	CKroneckerMean();
	CKroneckerMean(MatrixXd& Y, MatrixXd& weights, MatrixXd& fixedEffects, MatrixXd& A);
	virtual ~CKroneckerMean();

	virtual void setA(const MatrixXd& A);
	virtual void agetA(MatrixXd* out)
	{
		//generate the pseudo Design matrix that is equivalent ot the one used by CKroneckerMean
		//TODO: think about implementing CLinearMean as a special case of CKroneckerMean?!
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
typedef sptr<CKroneckerMean> PKroneckerMean;

inline void CKroneckerMean::checkDimensions(const MatrixXd& Y, const bool checkStrictWeights = true) const throw (CGPMixException)
{
	if (Y.rows() != this->fixedEffects.rows() && (muint_t)Y.cols() != this->getNTargets())
	{
		std::ostringstream os;
		os << this->getName() << ": Number of number samples and number of targets specified do not match with given Y. Y.rows() = " << Y.rows() << ", Y.cols() = " << Y.cols() << ", nSamples = " << fixedEffects.rows() << ", nTargets = " << this->getNTargets();
		throw gpmix::CGPMixException(os.str());
	}
	//this->checkDimensions(this->fixedEffects, this->weights, this->A, true, checkStrictWeights, true);
}

} /* namespace gpmix */
#endif /* CKRONECKERMEAN_H_ */
