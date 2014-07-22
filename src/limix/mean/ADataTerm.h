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

#ifndef ADATATERM_H_
#define ADATATERM_H_

#include "limix/types.h"

namespace limix {

class ADataTerm : public CParamObject{
protected:
	MatrixXd Y;

public:
	ADataTerm();
	ADataTerm(const MatrixXd& Y);
	virtual ~ADataTerm();
	virtual inline void setParams(const MatrixXd& params)
	{
		propagateSync(false);
	};

	virtual void aGetParams(MatrixXd* outParams){};
	virtual inline MatrixXd getParams(){ MatrixXd outParams = MatrixXd(); aGetParams(&outParams); return outParams;	};
	virtual inline void setY(const MatrixXd& Y)
	{
		checkDimensions(Y);
		this->Y = Y;
		propagateSync(false);

	}

	//getparams
	virtual muint_t getRowsParams() = 0;
	virtual muint_t getColsParams() = 0;

	virtual void aEvaluate(MatrixXd* Y);
	virtual void aGradY(MatrixXd* gradY);
	virtual void aGradParams(MatrixXd* outGradParamsRows, const MatrixXd* KinvY);
	virtual void aSumJacobianGradParams(MatrixXd* sumJacobianGradParams);
	virtual void aSumLogJacobian(MatrixXd* sumJacobianGradParams);

	virtual inline MatrixXd getY(){return Y;}
	virtual inline MatrixXd evaluate() { MatrixXd ret = MatrixXd(); aEvaluate(&ret); return ret;};
	virtual inline MatrixXd gradY() { MatrixXd ret = MatrixXd(); aGradY(&ret); return ret;};
	virtual inline MatrixXd gradParams(const MatrixXd& KinvY){ MatrixXd ret = MatrixXd(); aGradParams(&ret, &KinvY); return ret;};
	virtual inline MatrixXd sumJacobianGradParams(){ MatrixXd ret = MatrixXd(); aSumJacobianGradParams(&ret); return ret;};
	virtual inline MatrixXd sumLogJacobian(){ MatrixXd ret = MatrixXd(); aSumLogJacobian(&ret); return ret;};
	virtual inline std::string getName() const {return "ADataTerm";};
	virtual inline void checkDimensions(const MatrixXd& Y){};
};
typedef sptr<ADataTerm> PDataTerm;



} /* namespace limix */
#endif /* ADATATERM_H_ */
