/*
 * CGPopt.h
 *
 *  Created on: Jan 1, 2012
 *      Author: stegle
 */

#ifndef CGPOPT_H_
#define CGPOPT_H_

#include "gp_base.h";
#include "nlopt/api/nlopt.h"

namespace gpmix {

class CGPopt {
protected:
	CGPbase& gp;
	CGPHyperParams params;
	CGPHyperParams filter;
	double tolerance;
	muint_t numEvaluations;

	//objective without and with gradients
	double objective(const VectorXd& paramArray);
	double objective(const VectorXd& paramArray,VectorXd* gradParamArray);
	//optimization interface for nlopt:
	static double gptop_nlopt_objective(unsigned n, const double *x, double *grad, void *my_func_data);
public:
	CGPopt(CGPbase& gp);
	virtual ~CGPopt();

	virtual void opt();

	//getter and setter
	CGPHyperParams getParams() const;
	void setParams(CGPHyperParams params);
	CGPHyperParams getFilter() const;
	void setFilter(CGPHyperParams filter);
	double getTolerance() const;
	void getTolerance(double tol = 1E-4);



};




} /* namespace gpmix */
#endif /* CGPOPT_H_ */
