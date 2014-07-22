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

#include "se.h"
#include <math.h>
#include "limix/utils/matrix_helper.h"
#include "dist.h"

namespace limix {


CCovSqexpARD::~CCovSqexpARD() {
	// TODO Auto-generated destructor stub
}


void CCovSqexpARD::setNumberDimensions(muint_t numberDimensions)
{
	this->numberDimensions = numberDimensions;
	this->numberParams = numberDimensions+1;
}


void CCovSqexpARD::aKcross(MatrixXd* out, const CovarInput& Xstar ) const 
{
	//lengthscales
	MatrixXd L = params.block(1,0,params.rows()-1,1);
	//rescale with length
	MatrixXd x1l = Xstar * L.asDiagonal().inverse();
	MatrixXd x2l = this->X * L.asDiagonal().inverse();
	//squared exponential distance
	MatrixXd RV;
	sq_dist(&RV,x1l,x2l);
	RV*= -0.5;
	(*out) = std::pow(params(0),2)*RV.unaryExpr(std::ptr_fun(exp));
} // end :: K

void CCovSqexpARD::aKcross_diag(VectorXd* out, const CovarInput& Xstar) const 
{
    (*out) = params(0)*VectorXd::Ones(Xstar.rows());
}


void CCovSqexpARD::aKgrad_param(MatrixXd* out,const muint_t i) const 
{
	//lengthscales
	MatrixXd L = params.block(1,0,params.rows()-1,1);
	//rescale with length
	MatrixXd x1l = X * L.asDiagonal().inverse();
	//squared exponential distance
	MatrixXd RV;
	sq_dist(&RV,x1l,x1l);
	RV*= -0.5;
	(*out) = RV.unaryExpr(std::ptr_fun(exp));

	if (i==0) {
		(*out)*=2.*params(0);
	}
	else if (i<getNumberParams())
	{
		//lengthscale derivative:
		//1. get col (i-1) from X:
		MatrixXd sq;
		sq_dist(&sq,X.col(i-1),X.col(i-1));
		//3. elementwise product
        (*out)*=std::pow(params(0),2)*pow(params(i),-3);
		(*out).array()*=sq.array();
	}
	else
	{
		throw CLimixException("Parameter outside range");
	}
}
   
    
void CCovSqexpARD::aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const 
{

    muint_t i0, j0;
    
    if (i>j)    {i0=i; j0=j;}
    else        {muint_t temp=i; i0=j; j0=temp;}
    
    if ((i0==0) && (j0==0)) {
    	MatrixXd L = params.block(1,0,params.rows()-1,1);
    	MatrixXd x1l = X * L.asDiagonal().inverse();
    	MatrixXd RV;
    	sq_dist(&RV,x1l,x1l);
    	RV*= -0.5;
    	(*out) = 2*RV.unaryExpr(std::ptr_fun(exp));
    }
    else if ((j0==0) && (i0<getNumberParams()))
    {
        this->aKgrad_param(out,0);
        MatrixXd sq;
		sq_dist(&sq,X.col(i0-1),X.col(i0-1));
        (*out)*=pow(params(i0),-3);
		(*out).array()*=sq.array();
    }
    else if ((i0<getNumberParams()) && (j0!=i0))
    {
        this->aK(out);
        MatrixXd sq;
		sq_dist(&sq,X.col(i0-1),X.col(i0-1));
        (*out).array()*=sq.array();
        sq_dist(&sq,X.col(j0-1),X.col(j0-1));
        (*out).array()*=sq.array();
        (*out)*=pow(params(i0),-3)*pow(params(j0),-3);
    }
    else if ((i0<getNumberParams()) && (j0==i0))
    {
        //this->aK(out);
        MatrixXd sq;
		sq_dist(&sq,X.col(i0-1),X.col(i0-1));
        //(*out)=pow(params(i0),-6)*sq.unaryExpr(std::bind1st(std::ptr_fun(pow), 2))-3.0*pow(params(i0),-4)*sq;
        (*out)=sq;
        (*out).array()*=sq.array();
        (*out)*=pow(params(i0),-6);
        (*out)=(*out)-3*pow(params(i0),-4)*sq;
        (*out).array()*=K().array();
    }
    else
    {
        throw CLimixException("Parameter outside range");
    }
}


void CCovSqexpARD::aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const 
{
    //Check d not outside range
    if (d>=(muint_t)this->numberDimensions) throw CLimixException("Dimension outside range");
    
    this->aKcross(out,Xstar);
    //lengthscales: now we need to squre explicitly
    MatrixXd dist;
    lin_dist(&dist,X,Xstar,d);
    //rescale with squared lengthscale of the corresponding dimension:
    dist/=(-1.0*pow(params(d+1),2));
    //pointwise product with out array
    (*out).array() *= dist.array();
}

void CCovSqexpARD::aKdiag_grad_X(VectorXd* out,const muint_t d) const 
{
	(*out) = VectorXd::Zero(X.rows());
}




} /* namespace limix */
