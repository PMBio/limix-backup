/*
 * CGPopt.cpp
 *
 *  Created on: Jan 1, 2012
 *      Author: stegle
 */

#include "gp_opt.h"
#include <iostream>
#include <limits>

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


void CGPopt::completeConstraints(CGPHyperParams& constraints, const CGPHyperParams& params,mfloat_t fill_value)
{
	for(CGPHyperParamsMap::const_iterator iter = params.begin(); iter!=params.end();iter++)
		{
			//1. get param object which can be a matrix or array:
			string name = (*iter).first;
			MatrixXd value = (*iter).second;
			if (!constraints.exists(name))
			{
				MatrixXd value_fill =value;
				value_fill.fill(fill_value);
				constraints[name] = value_fill;
			}
		}
}



void CGPopt::opt() throw (CGPMixException)
{
	//0. set evaluation counter to 0:
	numEvaluations = 0;
	//1. get starging point
	CGPHyperParams params = gp.getParams();
	VectorXd x = params.getParamArray();

	muint_t numParams = x.rows();
	//2. create optimization instance:
	optimizer = nlopt_create(solver, numParams);
	nlopt_set_min_objective(optimizer, CGPopt::gpopt_nlopt_objective, this);
	//3. set tolerance
	nlopt_set_xtol_rel(optimizer, tolerance);
	//4. set constraints
	VectorXd x_min;
	VectorXd x_max;
	if ((optBoundLower.size()>0) || (optBoundUpper.size()>0))
	{
		//1. complete constraints
		completeConstraints(optBoundUpper,params,+1.0*std::numeric_limits<mfloat_t>::infinity());
		completeConstraints(optBoundLower,params,-1.0*std::numeric_limits<mfloat_t>::infinity());
		//2. get constraints
		x_min = optBoundLower.getParamArray();
		x_max = optBoundUpper.getParamArray();
		//3. check that they have the same shape than X
		if ((x_min.rows()!=x_max.rows()) || (x.rows()!=x_max.rows()))
		{
			throw CGPMixException("Constraints and parameters of gp optimization have incompatible shape.");
		}
		nlopt_set_lower_bounds(optimizer, x_min.data());
		nlopt_set_upper_bounds(optimizer, x_max.data());
	}
	double* x0d = x.data();
	double minf; /* the minimum objective value, upon return */
	if (nlopt_optimize(optimizer, x0d, &minf) < 0) {
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
	for (muint_t i=0;i<(muint_t)((x0.rows()));i++){
            mfloat_t change = relchange * x0(i);
            change = max(change, 1E-5);
            x(i) = x0(i) + change;
            gp.setParamArray(x);
            mfloat_t Lplus = gp.LML();
            x(i) = x0(i) - change;
            gp.setParamArray(x);
            mfloat_t Lminus = gp.LML();
            //restore
            x(i) = x0(i);
            //numerical gradient
            mfloat_t diff_numerical = (Lplus - Lminus) / (2. * change);
            grad_numerical(i) = diff_numerical;
        }
        std::cout << grad_numerical << "\n";
        std::cout << grad_analyt << "\n";
        std::cout << (grad_numerical - grad_analyt) << "\n";
        rv = ((grad_numerical - grad_analyt).squaredNorm() < threshold);
        return rv;
    }

    void CGPopt::setFilter(CGPHyperParams filter)
    {
        this->filter = filter;
    }

    CGPopt::~CGPopt()
    {
        // TODO Auto-generated destructor stub
    }

    mfloat_t CGPopt::objective(const VectorXd & paramArray)
    {
        this->numEvaluations++;
        double lml;
        //set Params
        gp.setParamArray(paramArray);
        lml = gp.LML();
        return lml;
    }

    mfloat_t CGPopt::objective(const VectorXd & paramArray, VectorXd *gradParamArray)
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

    CGPHyperParams CGPopt::getOptBoundLower() const
    {
        return optBoundLower;
    }

    CGPHyperParams CGPopt::getOptBoundUpper() const
    {
        return optBoundUpper;
    }

    void CGPopt::setOptBoundUpper(CGPHyperParams optBoundUpper)
    {
        this->optBoundUpper = optBoundUpper;
    }

    void CGPopt::setOptBoundLower(CGPHyperParams optBoundLower)
    {
        this->optBoundLower = optBoundLower;
    }

    void CGPopt::setTolerance(mfloat_t tol)
    {
        this->tolerance = tol;
    }

    double CGPopt::gpopt_nlopt_objective(unsigned  n, const double *x, double *grad, void *my_func_data)
    {
        double lml;
        //1. cast additional data as Gptop
        CGPopt *gpopt = (CGPopt*)(((my_func_data)));
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
