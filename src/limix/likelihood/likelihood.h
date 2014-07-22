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

#ifndef LIKELIHOOD_H_
#define LIKELIHOOD_H_

#include "limix/covar/covariance.h"

namespace limix {


typedef VectorXd LikParams;


//For now, likelihood is a special case of covariance function
class ALikelihood : public ACovarianceFunction {
	//indicator if the class is synced with the cache
protected:
public:
	ALikelihood(const muint_t numberParams=1);
	virtual ~ALikelihood();

	//pure virtual functions we don't really need...
	virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const ;
	void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const ;
	virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const ;
	virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const ;
};
typedef sptr<ALikelihood> PLikelihood;



/* Null likelihood model: assuming all variation is explained in covar*/
class CLikNormalNULL : public ALikelihood {
protected:
	muint_t numRows;
public:
	CLikNormalNULL();
	~CLikNormalNULL();

	//pure virtual functions that need to be overwritten
	virtual void aK(MatrixXd* out) const ;
	virtual void aKdiag(VectorXd* out) const ;
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const ;
	virtual void aKgrad_param(MatrixXd* out, const muint_t row) const ;
    virtual void aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const ;
	//overwrite setX. We merely ignore the number of columns here:
	virtual void setX(const CovarInput& X) ;

	std::string getName() const {return "LikNormalIso";};
};
typedef sptr<CLikNormalNULL> PLikNormalNULL;


/* Gaussian likelihood model*/
class CLikNormalIso : public ALikelihood {
protected:
	muint_t numRows;
public:
	CLikNormalIso();
	~CLikNormalIso();

	//pure virtual functions that need to be overwritten
	virtual void aK(MatrixXd* out) const ;
	virtual void aKdiag(VectorXd* out) const ;
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const ;
	virtual void aKgrad_param(MatrixXd* out, const muint_t row) const ;
    virtual void aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const ;
	//overwrite setX. We merely ignore the number of columns here:
	virtual void setX(const CovarInput& X) ;

	std::string getName() const {return "LikNormalIso";};
};
typedef sptr<CLikNormalIso> PLikNormalIso;

/* Likelihood model for SVD Covariances */
class CLikNormalSVD : public ALikelihood {
protected:
	muint_t numRows;
public:
	CLikNormalSVD();
	~CLikNormalSVD();

	//pure virtual functions that need to be overwritten
	virtual void aK(MatrixXd* out) const ;
	virtual void aKdiag(VectorXd* out) const ;
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const ;
	virtual void aKgrad_param(MatrixXd* out, const muint_t row) const ;
    virtual void aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const ;
	//overwrite setX. We merely ignore the number of columns here:
	virtual void setX(const CovarInput& X) ;

	virtual mfloat_t getSigmaK2();
	virtual mfloat_t getDelta();
	virtual mfloat_t getSigmaK2grad();
	virtual mfloat_t getDeltagrad();

	std::string getName() const {return "ClikNormalSVD";};
};
typedef sptr<CLikNormalSVD> PLikNormalSVD;




} /* namespace limix */
#endif /* LIKELIHOOD_H_ */
