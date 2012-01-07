/*
 * CGPopt.cpp
 *
 *  Created on: Jan 1, 2012
 *      Author: stegle
 */

#include "gp_opt.h"
#include <iostream>

namespace gpmix {

CGPopt::CGPopt(CGPbase& gp) : gp (gp)
{
	tolerance = DEFAULT_TOL;
	// TODO Auto-generated constructor stub
}


CGPHyperParams CGPopt::getFilter() const
{
	return filter;
}

void CGPopt::opt()
{
	//0. set evaluation counter to 0:
	numEvaluations = 0;
	//1. get starging point
	VectorXd x = gp.getParamArray();
	muint_t numParams = x.rows();
	//2. create optimization instance:
	nlopt_opt opt = nlopt_create(solver, numParams);
	nlopt_set_min_objective(opt, CGPopt::gpopt_nlopt_objective, this);
	//3. set tolerance
	nlopt_set_xtol_rel(opt, tolerance);
	//4. set constraints
	//TODO
	//nlopt_add_inequality_constraint(opt, myconstraint, &data[0], 1e-8);

	double* x0d = x.data();
	double minf; /* the minimum objective value, upon return */
	if (nlopt_optimize(opt, x0d, &minf) < 0) {
	    std::cout << "nlopt failed!\n";
	}
	else {
		//1. reevaluate at optimum
		VectorXd df;
		/*
		double lml_opt = objective(x,&df);
		//2. diagonoses:

		std::cout << "Optimum found for: f(x=["<<x<<"]) = "<< lml_opt << "\n";
	    std::cout << "df(x)=[" << df << "]\n";
	    std::cout << "Function evaluations: " << numEvaluations << "\n";
	    std::cout << "----------" << "\n";
	    */
	    //store optimized values:
	    optParams = gp.getParams();
	    optParams.setParamArray(x);
	    // dump gradients:
	    std::cout << "covar grad:" << gp.LMLgrad()["covar"] << "\n";
	    std::cout << "lml:" << gp.LML() << "\n";
	}
}


bool CGPopt::gradCheck(mfloat_t relchange,mfloat_t threshold)
{
	bool rv;
	//current x0:
	VectorXd x0 = gp.getParamArray();
	VectorXd x  = x0;

	//1. analytical solution
	VectorXd grad_analyt;
	gp.aLMLgrad(&grad_analyt);
	//2. numerical solution;
	VectorXd grad_numerical(grad_analyt.rows());
	//loop
	for (muint_t i=0;i<(muint_t)x0.rows();i++)
	{
		mfloat_t change = relchange*x0(i);
		change = max(change,1E-5);
		x(i) = x0(i) + change;
		gp.setParamArray(x);
		mfloat_t Lplus = gp.LML();
		x(i) = x0(i) - change;
		gp.setParamArray(x);
		mfloat_t Lminus = gp.LML();
		//restore
		x(i) = x0(i);
		//numerical gradient
		mfloat_t diff_numerical  = (Lplus-Lminus)/(2.*change);
		grad_numerical(i) = diff_numerical;
	}
	rv = ((grad_numerical-grad_analyt).squaredNorm()<threshold);
	return rv;
}

void CGPopt::setFilter(CGPHyperParams filter)
{
	this->filter = filter;}

CGPopt::~CGPopt() {
	// TODO Auto-generated destructor stub
}

mfloat_t CGPopt::objective(const VectorXd& paramArray)
{
	this->numEvaluations++;
	double lml;
	//set Params
	gp.setParamArray(paramArray);
	lml = gp.LML();
	return lml;
}


mfloat_t CGPopt::objective(const VectorXd& paramArray,VectorXd* gradParamArray)
{
	this->numEvaluations++;
	double lml;
	//set Params
	gp.setParamArray(paramArray);
	lml = gp.LML();
	gp.aLMLgrad(gradParamArray);
	return lml;
}

mfloat_t CGPopt::getTolerance() const
    {
        return tolerance;
    }

void CGPopt::setTolerance(mfloat_t tol)
    {
        this->tolerance = tol;
   }

double CGPopt::gpopt_nlopt_objective(unsigned  n, const double *x, double *grad, void *my_func_data)
{
    	double lml;
        //1. cast additional data as Gptop
        CGPopt *gpopt = (CGPopt*)(my_func_data);
        //2. map parameters and gradients
        Eigen::Map<const VectorXd > vX(x,n);

        //need gradient?
        if (grad)
        {
        	VectorXd vgrad;
        	lml = gpopt->objective(vX,&vgrad);
        	//copy:
        	Eigen::Map<VectorXd > vG(grad,n);
        	vG = vgrad;
        }
        else
        	lml = gpopt->objective(vX);
        //std::cout << "f("<< vX << ")="<< lml << "\n\n";
        return lml;
    }


} /* namespace gpmix */
