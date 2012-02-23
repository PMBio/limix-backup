/*
 * CSumLinear.h
 *
 *  Created on: Jan 23, 2012
 *      Author: clippert
 */

#ifndef CSUMLINEAR_H_
#define CSUMLINEAR_H_

#include "ADataTerm.h"
#include "CLinearMean.h"
#include <vector>

namespace gpmix {

#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//%shared_ptr(gpmix::CSumLinear)
#endif

typedef std::vector<PLinearMean> VecLinearMean;
class CSumLinear: public gpmix::ADataTerm {
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


} /* namespace gpmix */
#endif /* CSUMLINEAR_H_ */

