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

#include "gp_opt.h"
#include <iostream>
#include <limits>

namespace limix {

CGPopt::CGPopt(PGPbase gp) : gp (gp)
{
	tolerance = DEFAULT_TOL;
	// TODO Auto-generated constructor stub
}


CGPHyperParams CGPopt::getParamMask() const
{
	return optParamMask;
}


void CGPopt::completeConstraints(CGPHyperParams& constraints, const CGPHyperParams& params,mfloat_t fill_value)
{
	for(CGPHyperParams::const_iterator iter = params.begin(); iter!=params.end();iter++)
		{
			//1. get param object which can be a matrix or array:
			std::string name = (*iter).first;
			MatrixXd value = (*iter).second;
			if (!constraints.exists(name))
			{
				MatrixXd value_fill =value;
				value_fill.fill(fill_value);
				constraints[name] = value_fill;
			}
		}
}



bool CGPopt::opt() 
{
	//0. set evaluation counter to 0:
	numEvaluations = 0;
	//1. get parameter vector from GP object
	CGPHyperParams params0 = gp->getParams();

	//do we have a parameter mask?
	if(optParamMask.size()==0)
	{
		//no?
		//get from GP object
		optParamMask = gp->getParamMask();
	}

	//1.2 determine effective number of parameters we optimize over:
	muint_t numOptParams = params0.getNumberParams(optParamMask);

	//2. create optimization instance:
	optimizer = nlopt_create(solver, numOptParams);
	nlopt_set_min_objective(optimizer, CGPopt::gpopt_nlopt_objective, this);
	//3. set tolerance
	nlopt_set_xtol_rel(optimizer, tolerance);
	//4. set constraints
	VectorXd x_min;
	VectorXd x_max;
	CGPHyperParams optBoundLower_;
	CGPHyperParams optBoundUpper_;

	//are user-defined constraints set?
	if ((optBoundLower.size()==0) && (optBoundUpper.size()==0))
	{
		//no?
		//query gpbase object
		optBoundUpper_ = CGPHyperParams(gp->getParamBounds(true));
		optBoundLower_ = CGPHyperParams(gp->getParamBounds(false));
	}
	else
	{
		//else: get user constraints and create a copy
		optBoundLower_ = CGPHyperParams(optBoundLower);
		optBoundUpper_ = CGPHyperParams(optBoundUpper);
	}
	// complete constraints
	completeConstraints(optBoundUpper_,params0,+1.0*std::numeric_limits<mfloat_t>::infinity());
	completeConstraints(optBoundLower_,params0,-1.0*std::numeric_limits<mfloat_t>::infinity());
	// mask out relevant parts
	x_min = optBoundLower_.getParamArray(optParamMask);
	x_max = optBoundUpper_.getParamArray(optParamMask);
	//3. check that they have the same shape than X
	if ((x_min.rows()!=x_max.rows()) || (numOptParams!=(muint_t)x_max.rows()))
	{
			throw CLimixException("Constraints and parameters of gp optimization have incompatible shape.");
	}
	nlopt_set_lower_bounds(optimizer, x_min.data());
	nlopt_set_upper_bounds(optimizer, x_max.data());


	double minf; /* the minimum objective value, upon return */
	std::vector<CGPHyperParams> opt_start_points;

	//5. is starting point list empty? if yes: use current parameters (a single starting point)
	if(this->optStartParams.size()==0)
		opt_start_points.push_back(params0);
	else
		opt_start_points = this->optStartParams;


	//6. loop over starting points for optimization and run:
	//param array of solutions
	MatrixXd optParamArray = MatrixXd(opt_start_points.size(),numOptParams);
	VectorXd optLMLArray   = VectorXd(opt_start_points.size());
	optLMLArray.setConstant(HUGE_VAL);
	muint_t iopt_success =0;
	for(std::vector<CGPHyperParams>::const_iterator iter = opt_start_points.begin(); iter!=opt_start_points.end();iter++)
	{
		//get starting point
		const CGPHyperParams& params0_ = (*iter);
		//array version, filtered
		VectorXd paramsArray0_ = params0_.getParamArray(optParamMask);
		//std::cout << "\n" << paramsArray0_ << "\n"<< "\n";
		double* x0d = paramsArray0_.data();
		double optret = nlopt_optimize(optimizer, x0d, &minf);
		if (optret < 0)
		{
		    std::cout << "nlopt failed!\n";
		}
		else
		{
			//1. reevaluate at optimum
		    //store optimized values:
			CGPHyperParams _optParams = gp->getParams();
			optParamArray.row(iopt_success) = _optParams.getParamArray(optParamMask);
			optLMLArray(iopt_success) = gp->LML();
			iopt_success++;
		}
	}
	//get best solution in restart array
	VectorXd::Index argmax;
	//std::cout << "LML " << optLMLArray << "\n";
	optLML = optLMLArray.minCoeff(&argmax);
	//get corresponding param array and store
	optParams = params0;
	optParams.setParamArray(optParamArray.row(argmax),optParamMask);

	return (iopt_success>0);
}


bool CGPopt::gradCheck(mfloat_t relchange,mfloat_t threshold)
{
	bool rv;
	//current x0:
	VectorXd x0 = gp->getParamArray();
	VectorXd x  = x0;

	//1. analytical solution
	VectorXd grad_analyt;
	gp->aLMLgrad(&grad_analyt);
	//2. numerical solution;
	VectorXd grad_numerical(grad_analyt.rows());
	//loop
	for (muint_t i=0;i<(muint_t)(((x0.rows())));i++){
            mfloat_t change = relchange * x0(i);
            change = std::max(change, 1E-5);
            x(i) = x0(i) + change;
            gp->setParamArray(x);
            mfloat_t Lplus = gp->LML();
            x(i) = x0(i) - change;
            gp->setParamArray(x);
            mfloat_t Lminus = gp->LML();
            //restore
            x(i) = x0(i);
            //numerical gradient
            mfloat_t diff_numerical = (Lplus - Lminus) / (2. * change);
            grad_numerical(i) = diff_numerical;
        }
        std::cout <<"numerical:\n"<< grad_numerical << "\n";
        std::cout <<"analytical:\n"<< grad_analyt << "\n";
        std::cout <<"diff:\n"<< (grad_numerical - grad_analyt) << "\n";
        rv = ((grad_numerical - grad_analyt).squaredNorm() < threshold);
        std::cout <<"passed?:\n"<< rv << "\n";
        return rv;
    }

    void CGPopt::setParamMask(CGPHyperParams filter)
    {
        this->optParamMask = filter;
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
        gp->setParamArray(paramArray,optParamMask);
        lml = gp->LML();
        return lml;
    }

    mfloat_t CGPopt::objective(const VectorXd & paramArray, VectorXd *gradParamArray)
    {
        this->numEvaluations++;
        double lml;
        //set Params
        gp->setParamArray(paramArray,optParamMask);
        //std::cout << "D:" << gp.getParamArray().segment(1,paramArray.rows())-paramArray << "\n\n";
        //std::cout << paramArray << "--" << gp.getParamArray() << "\n\n";
        lml = gp->LML();
        CGPHyperParams grad = gp->LMLgrad();
        //std::cout << "dLML(" << grad << ")" << "\n";
        grad.agetParamArray(gradParamArray,optParamMask);
        //std::cout << "Dgrad" << grad.getParamArray().segment(1,paramArray.rows())-*gradParamArray << "\n\n";
        //std::cout << "dLMLmask(" << grad << ")" << "\n";
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

    std::vector<CGPHyperParams> CGPopt::getOptStartParams() const
    {
        return optStartParams;
    }

    void CGPopt::setOptStartParams(const std::vector<CGPHyperParams>& optStartParams)
    {
        this->optStartParams = optStartParams;
    }

    void CGPopt::addOptStartParams(const CGPHyperParams& params)
    {
    	this->optStartParams.push_back(params);

    }
    void CGPopt::addOptStartParams(const VectorXd& paramArray)
    {
    	//1. convert to hyperParams object
    	CGPHyperParams params(this->gp->getParams());
    	params.setParamArray(paramArray);
    	addOptStartParams(params);
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
        CGPopt *gpopt = (CGPopt*)((((my_func_data))));
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


} /* namespace limix */
