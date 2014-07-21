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

#ifndef CSUMLINEAR_H_
#define CSUMLINEAR_H_

#include "ADataTerm.h"
#include "CLinearMean.h"
#include <vector>

namespace limix {

typedef std::vector<PLinearMean> VecLinearMean;
class CSumLinear: public ADataTerm {
	VecLinearMean terms;
	size_t nParams;
public:
	CSumLinear();
	virtual ~CSumLinear();
	virtual void aGetParams(MatrixXd* outParams);
	virtual void setParams(const MatrixXd& params);
	virtual void aEvaluate(MatrixXd* Y);
	virtual void aGradParams(MatrixXd* outGradParams, const MatrixXd* KinvY);
	virtual inline void appendTerm(PLinearMean term) { this->terms.push_back(term);	propagateSync(false);
 };
	virtual inline PLinearMean getTerm(muint_t ind) { return terms[ind]; };
	virtual inline muint_t getNterms() const { return (muint_t) terms.size(); };
	virtual inline VecLinearMean& getTerms() { return terms;};


	//getparams
	virtual muint_t getRowsParams();
	virtual muint_t getColsParams();

};

typedef sptr<CSumLinear> PSumLinear;


} /* namespace limix */
#endif /* CSUMLINEAR_H_ */

