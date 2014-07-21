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

#ifndef LINEAR_H_
#define LINEAR_H_

#include "limix/covar/covariance.h"

namespace limix {

class CCovLinearISO: public ACovarianceFunction  {
public:
	CCovLinearISO(muint_t numberDimensions=1) : ACovarianceFunction(1)
	{
		this->setNumberDimensions(numberDimensions);
		initParams();
	}

	~CCovLinearISO();

	//overloaded pure virtual functions:
	void aKcross(MatrixXd* out, const CovarInput& Xstar ) const ;
	void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const ;

	void aKgrad_param(MatrixXd* out,const muint_t i) const ;
    void aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const ;
	void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const ;
	void aKdiag_grad_X(VectorXd* out,const muint_t d) const ;

	//class information
	inline std::string getName() const {return "CCovLinearISO";};

};
typedef sptr<CCovLinearISO> PCovLinearISO;

class CCovLinearARD: public ACovarianceFunction  {
public:
	CCovLinearARD(muint_t numberDimensions=1) : ACovarianceFunction(1)
	{
		this->setNumberDimensions(numberDimensions);
		initParams();
	}

	~CCovLinearARD();
	//overloaded virtuals
	virtual void setNumberDimensions(muint_t numberDimensions);

	//overloaded pure virtual functions:
	void aKcross(MatrixXd* out, const CovarInput& Xstar ) const ;
	void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const ;
	void aKgrad_param(MatrixXd* out,const muint_t i) const ;
    void aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const ;
	void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const ;
	void aKdiag_grad_X(VectorXd* out,const muint_t d) const ;

	//class information
	inline std::string getName() const{ return "CovLinearARD";}
};
typedef sptr<CCovLinearARD> PCovLinearARD;



/* Delta kernel */
class CCovLinearISODelta: public ACovarianceFunction  {
public:
	CCovLinearISODelta(muint_t numberDimensions=1) : ACovarianceFunction(1)
	{
		this->setNumberDimensions(numberDimensions);
		initParams();
	}

	~CCovLinearISODelta();

	//overloaded pure virtual functions:
	void aKcross(MatrixXd* out, const CovarInput& Xstar ) const ;
	void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const ;

	void aKgrad_param(MatrixXd* out,const muint_t i) const ;
    void aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const ;


	//class information
	inline std::string getName() const {return "CCovLinearISODelta";};

};
typedef sptr<CCovLinearISODelta> PCovLinearISODelta;




} /* namespace limix */
#endif /* LINEAR_H_ */
