/*
 * CLinearMean.h
 *
 *  Created on: Jan 3, 2012
 *      Author: clippert
 */

#ifndef CLINEARMEAN_H_
#define CLINEARMEAN_H_

#include "ADataTerm.h"
#include "gpmix/utils/matrix_helper.h"

namespace gpmix {

class CLinearMean: public ADataTerm {
	MatrixXd weights;
	MatrixXd fixedEffects;
	muint_t nTargets;
	void zeroInitWeights();
public:
	CLinearMean();
	CLinearMean(muint_t nTargets);
	CLinearMean(MatrixXd& Y, MatrixXd& weights, MatrixXd& fixedEffects);
	CLinearMean(MatrixXd& Y, MatrixXd& fixedEffects);
	virtual ~CLinearMean();

	virtual void aEvaluate(MatrixXd* outY);
	void aGradParams(MatrixXd* outGradParams);

	virtual void setParams(MatrixXd& weightMatrix);
	virtual void setFixedEffects(MatrixXd& fixedEffects);

	virtual void aGetParams(MatrixXd* outParams);
	virtual void aGetFixedEffects(MatrixXd* outFixedEffects);
	virtual void aPredictY(MatrixXd* outY) const ;
	virtual void aPredictYstar(MatrixXd* outY, const MatrixXd* fixedEffects) const;

	virtual inline MatrixXd getFixedEffects(){MatrixXd outFixedEffects; this->aGetFixedEffects(&outFixedEffects); return outFixedEffects;}

	virtual inline string getName() const {return "CLinearMean";};
	virtual inline void checkDimensions(const MatrixXd& Y){checkDimensions(this->weights, this->fixedEffects, Y, false, false, true);};
	virtual inline void checkDimensions(const MatrixXd& weights, const MatrixXd& fixedEffects, const MatrixXd& Y, const bool checkStrictWeights = false, const bool checkStrictFixedEffects = false, const bool checkStrictY = false) const throw (CGPMixException);
	inline virtual MatrixXd predictY() const {MatrixXd out = MatrixXd(); aPredictY(&out); return out;};
	inline virtual MatrixXd predictY(const MatrixXd& fixedEffects) const {MatrixXd out = MatrixXd(); aPredictYstar(&out, &fixedEffects); return out;};
	virtual void setWeightsOLS();
	virtual void setWeightsOLS(const MatrixXd& Y);
	virtual inline muint_t getNTargets() const {return nTargets;}
};

inline void CLinearMean::checkDimensions(const MatrixXd& weights, const MatrixXd& fixedEffects, const MatrixXd& Y, const bool checkStrictWeights, const bool checkStrictFixedEffects, const bool checkStrictY) const throw (CGPMixException)
{
	bool notIsnullweights = false;
	bool notIsnullFixed = false;
	bool notIsnullY = false;

	//if no strict testing is performed, we test, if the argument has been set
	if(!checkStrictWeights)
	{
		notIsnullweights = !isnull(weights);
	}
	if(!checkStrictFixedEffects)
	{
		notIsnullFixed = !isnull(fixedEffects);
	}
	if(!checkStrictY)
	{
		notIsnullY = !isnull(Y);
	}

	if (notIsnullweights && notIsnullFixed && (weights.rows()) != fixedEffects.cols() ){
		ostringstream os;
		os << this->getName() << ": Number of weights and fixed effects do not match. number fixed effects = " << fixedEffects.cols() << ", number weights = " << weights.rows();
		throw gpmix::CGPMixException(os.str());
	}
	if (notIsnullFixed && notIsnullY && (fixedEffects.rows()) != Y.rows() ){
			ostringstream os;
			os << this->getName() << ": Number of samples in fixedEffects and Y do not match. fixed effects : " << fixedEffects.rows() << ", Y = " << Y.rows();
			throw gpmix::CGPMixException(os.str());
		}
	if ( notIsnullweights && notIsnullY && (weights.cols()) != Y.cols() ){
			ostringstream os;
			os << this->getName() << ": Number of target dimensions do not match in Y and weights. Y: " << Y.cols() << ", weights = " << weights.cols();
			throw gpmix::CGPMixException(os.str());
		}
}


} /* namespace gpmix */
#endif /* CLINEARMEAN_H_ */
