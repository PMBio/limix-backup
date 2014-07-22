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

#ifndef CCOVSQEXPARD_H_
#define CCOVSQEXPARD_H_

#include "covariance.h"

namespace limix {

class CCovSqexpARD: public ACovarianceFunction {
public:
	CCovSqexpARD(muint_t numberDimensions=1): ACovarianceFunction(1)
	{
		this->setNumberDimensions(numberDimensions);
		initParams();
	}

	~CCovSqexpARD();

	virtual void setNumberDimensions(muint_t numberDimensions);


	//overloaded pure virtual functions:
	virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const ;
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const ;
	virtual void aKgrad_param(MatrixXd* out,const muint_t i) const ;
    virtual void aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const ;
	virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const ;
	virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const ;

	//class information
	inline std::string getName() const{ return "CovSEARD";}
}; //end class CCovSqexpARD

typedef sptr<CCovSqexpARD> PCovSqexpARD;


} /* namespace limix */
#endif /* CCOVSQEXPARD_H_ */
