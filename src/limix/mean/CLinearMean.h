// Copyright(c) 2014, The LIMIX developers(Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.

#ifndef CLINEARMEAN_H_
#define CLINEARMEAN_H_

#include "ADataTerm.h"
#include "limix/utils/matrix_helper.h"

namespace limix {

class CLinearMean: public ADataTerm {
	//friend class CKroneckerMean;
protected:
	MatrixXd weights;
	MatrixXd fixedEffects;
	muint_t nTargets;
	void zeroInitWeights();
public:
	CLinearMean();
	CLinearMean(muint_t nTargets);
	CLinearMean(const MatrixXd& Y, const MatrixXd& weights, const MatrixXd& fixedEffects);
	CLinearMean(const MatrixXd& Y, const MatrixXd& fixedEffects);
	virtual ~CLinearMean();

	virtual void agetA(MatrixXd* out)
	{
		//generate the pseudo Design matrix that is equivalent ot the one used by CKroneckerMean
		//TODO: think about implementing CLinearMean as a special case of CKroneckerMean?!
		(*out) = MatrixXd::Identity(nTargets,nTargets);
	}
	virtual MatrixXd getA()
	{
			//generate the pseudo Design matrix that is equivalent ot the one used by CKroneckerMean
			//TODO: think about implementing CLinearMean as a special case of CKroneckerMean?!
			MatrixXd rv;
			agetA(&rv);
			return rv;
	}


	virtual void aEvaluate(MatrixXd* outY);
	virtual void aGradParams(MatrixXd* outGradParams, const MatrixXd* KinvY);

	virtual void setParams(const MatrixXd& weightMatrix);
	virtual void setFixedEffects(const MatrixXd& fixedEffects);

	virtual void aGetParams(MatrixXd* outParams);
	virtual void aGetFixedEffects(MatrixXd* outFixedEffects);
	virtual void aPredictY(MatrixXd* outY) const ;
	virtual void aPredictYstar(MatrixXd* outY, const MatrixXd* fixedEffects) const;
	virtual inline muint_t getRowsParams() {return (muint_t) this->fixedEffects.cols();};
	virtual muint_t getColsParams() {
		return this->nTargets;
	}
	virtual inline MatrixXd getFixedEffects(){MatrixXd outFixedEffects; this->aGetFixedEffects(&outFixedEffects); return outFixedEffects;};
	virtual inline std::string getName() const {return "CLinearMean";};
	virtual inline void checkDimensions(const MatrixXd& Y){checkDimensions(this->weights, this->fixedEffects, Y, false, false, true);};
	virtual inline void checkDimensions(const MatrixXd& weights, const MatrixXd& fixedEffects, const MatrixXd& Y, const bool checkStrictWeights = false, const bool checkStrictFixedEffects = false, const bool checkStrictY = false) const ;
	inline virtual MatrixXd predictY() const {MatrixXd out = MatrixXd(); aPredictY(&out); return out;};
	inline virtual MatrixXd predictY(const MatrixXd& fixedEffects) const {MatrixXd out = MatrixXd(); aPredictYstar(&out, &fixedEffects); return out;};
	virtual void setWeightsOLS();
	virtual void setWeightsOLS(const MatrixXd& Y);
	virtual inline muint_t getNTargets() const {return nTargets;}
};

inline void CLinearMean::checkDimensions(const MatrixXd& weights, const MatrixXd& fixedEffects, const MatrixXd& Y, const bool checkStrictWeights, const bool checkStrictFixedEffects, const bool checkStrictY) const 
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
		std::ostringstream os;
		os << this->getName() << ": Number of weights and fixed effects do not match. number fixed effects = " << fixedEffects.cols() << ", number weights = " << weights.rows();
		throw CLimixException(os.str());
	}
	if (notIsnullFixed && notIsnullY && (fixedEffects.rows()) != Y.rows() ){
			std::ostringstream os;
			os << this->getName() << ": Number of samples in fixedEffects and Y do not match. fixed effects : " << fixedEffects.rows() << ", Y = " << Y.rows();
			throw CLimixException(os.str());
		}
	if ( notIsnullweights && notIsnullY && (weights.cols()) != Y.cols() ){
			std::ostringstream os;
			os << this->getName() << ": Number of target dimensions do not match in Y and weights. Y: " << Y.cols() << ", weights = " << weights.cols();
			throw CLimixException(os.str());
		}
}
typedef sptr<CLinearMean> PLinearMean;

} /* namespace limix */
#endif /* CLINEARMEAN_H_ */
