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
class CSumLinear: public gpmix::ADataTerm {
	std::vector<CLinearMean*> terms;
	size_t nParams;
public:
	CSumLinear();
	virtual ~CSumLinear();
	virtual void aGetParams(MatrixXd* outParams);
	virtual void setParams(const MatrixXd& params);
	virtual void aEvaluate(MatrixXd* Y);
	virtual void aGradParams(MatrixXd* outGradParams, const MatrixXd* KinvY);
	virtual inline void appendTerm(CLinearMean& term) { this->terms.push_back(&term); };
	virtual inline CLinearMean* getTerm(muint_t ind) { return terms[ind]; };
	virtual inline muint_t getNterms() const { return (muint_t) terms.size(); };
};

} /* namespace gpmix */
#endif /* CSUMLINEAR_H_ */

