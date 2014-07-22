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

#ifndef CKRONECKERMEAN_H_
#define CKRONECKERMEAN_H_

#include "ADataTerm.h"
#include "CLinearMean.h"
#include "limix/utils/matrix_helper.h"
namespace limix {


class CKroneckerMean : public CLinearMean {
protected:
	MatrixXd A;
public:
	CKroneckerMean();
	CKroneckerMean(MatrixXd& Y, MatrixXd& fixedEffects, MatrixXd& A);
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
	inline void checkDimensions(const MatrixXd& Y, const bool checkStrictWeights) const ;
	virtual inline std::string getName() const { return "CKoneckerFixedTerm"; };
	inline muint_t getDimFixedEffects() const { return this->fixedEffects.cols(); };
	virtual muint_t getColsParams()
	{
		return (muint_t) this->A.rows();
	}
};
typedef sptr<CKroneckerMean> PKroneckerMean;

inline void CKroneckerMean::checkDimensions(const MatrixXd& Y, const bool checkStrictWeights = true) const 
{
	if (Y.rows() != this->fixedEffects.rows() && (muint_t)Y.cols() != this->getNTargets())
	{
		std::ostringstream os;
		os << this->getName() << ": Number of number samples and number of targets specified do not match with given Y. Y.rows() = " << Y.rows() << ", Y.cols() = " << Y.cols() << ", nSamples = " << fixedEffects.rows() << ", nTargets = " << this->getNTargets();
		throw CLimixException(os.str());
	}
	//this->checkDimensions(this->fixedEffects, this->weights, this->A, true, checkStrictWeights, true);
}

} /* namespace limix */
#endif /* CKRONECKERMEAN_H_ */
