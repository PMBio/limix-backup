/*
 * trait.cpp
 *
 *  Created on: Jan 16, 2012
 *      Author: stegle
 */

#include "trait.h"
#include <math.h>
#include <cmath>
//#ifdef _WIN32
//#include <tgmath.h>
//#endif

namespace limix {

    
/* CTraitNew */
    
CTraitNew::CTraitNew(muint_t numberGroups)
{
    //1 input dimension which selects the group:
    this->numberDimensions = 1;
    //number of groups:
    this->numberGroups= numberGroups;
    this->X=MatrixXd::Zero(numberGroups,1);
}
    
CTraitNew::~CTraitNew()
{
}
    
muint_t CTraitNew::getNumberGroups() const
{return numberGroups;}

void CTraitNew::aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException)
{
	MatrixXd K;
	aKcross(&K,Xstar);
	(*out)=K.diagonal();
}
    
/* FREE FORM */
    
    
CTFreeFormNew::CTFreeFormNew(muint_t numberGroups) : CTraitNew(numberGroups)
{
    //number of parameters:
    this->numberParams = calcNumberParams(numberGroups);
}
    
muint_t CTFreeFormNew::calcNumberParams(muint_t numberGroups)
{
    return (0.5*numberGroups*(numberGroups+1));
}
    
CTFreeFormNew::~CTFreeFormNew()
{
}

void CTFreeFormNew::agetScales(CovarParams* out) {
    (*out) = this->params;
    double sign=1;
    if ((*out)(0)!=0) 	sign = std::abs((*out)(0))/((*out)(0));
    (*out)=sign*(*out);
    (*out)(this->numberParams-1)=std::abs((*out)(this->numberParams-1));
}
    
void CTFreeFormNew::aK0Covar2Params(VectorXd* out,const MatrixXd& K0)
{
}

void CTFreeFormNew::setParamsCovariance(const MatrixXd& K0) throw(CGPMixException)
{
    //0. check that the matrix has the correct size
    if(((muint_t)K0.rows()!=numberGroups) || ((muint_t)K0.cols()!=numberGroups))
    {
        throw CGPMixException("aK0Covar2Params: rows and columns need to be compatiable with the number of groups");
    }
    MatrixXd L;
    MatrixXdChol chol(K0);
    L =chol.matrixL();
    //2. create output argument and fill
    this->params.resize(this->numberParams);
    //3. loop over groups
    muint_t pindex=0;
    for(muint_t ir=0;ir<numberGroups;++ir)
        for (muint_t ic=0;ic<(ir+1);++ic)
        {
            params(pindex) = L(ir,ic);
            ++pindex;
        }
}

void CTFreeFormNew::aKcross(MatrixXd* out, const CovarInput& Xstar ) const  throw(CGPMixException)
{
    //create template matrix K
    MatrixXd L;
    agetL0(&L);
    (*out).noalias() = L*L.transpose();
}
    
void CTFreeFormNew::aKgrad_param(MatrixXd* out,const muint_t i) const throw(CGPMixException)
{
    MatrixXd L;
    MatrixXd Lgrad_parami;

    agetL0(&L);
    agetL0grad_param(&Lgrad_parami,i);

    //use chain rule K = LL^T
    (*out).noalias() = Lgrad_parami*L.transpose() + L*Lgrad_parami.transpose();
}


void CTFreeFormNew::aKhess_param(MatrixXd* out,const muint_t i,const muint_t j) const throw(CGPMixException)
{

    MatrixXd Lgrad_parami;
    MatrixXd Lgrad_paramj;

    agetL0grad_param(&Lgrad_parami,i);
    agetL0grad_param(&Lgrad_paramj,j);

    //use chain rule K = LL^T
    (*out).noalias() = Lgrad_parami*Lgrad_paramj.transpose() + Lgrad_paramj*Lgrad_parami.transpose();
}

void CTFreeFormNew::agetParamBounds0(CovarParams* lower,CovarParams* upper) const
{
    //all parameters but the diagonal elements are unbounded:
    *lower = -INFINITY*VectorXd::Ones(this->getNumberParams());
    *upper = +INFINITY*VectorXd::Ones(this->getNumberParams());
    //get diagonal elements
    //VectorXi isDiagonal =getIparamDiag();
    //set diagonal elements to be bounded [0,inf]
    //for (muint_t i=0;i<getNumberParams();++i)
    //{
    //    if(isDiagonal(i))
    //    {
    //        (*lower)(i) = 0;
    //    }
    //}
}
    
void CTFreeFormNew::agetParamMask0(CovarParams* out) const {
    (*out) = VectorXd::Ones(getNumberParams());
}
    
    
void CTFreeFormNew::setParamsVarCorr(const CovarParams& paramsVC) throw(CGPMixException)
{
    
    // No correlation must be <1 && >-1 or otherwise the initialization would fail
    for (muint_t i=numberGroups;i<getNumberParams();++i)
        if (paramsVC(i)>=1 || paramsVC(i)<=-1)
            throw CGPMixException("Correlation must be in (-1,+1)");
    
    //0. check that the matrix has the correct size
    checkParamDimensions(paramsVC);
    
    
    // It takes variances and correlations, builds K0 and then setParamsCovarianceK0
    
    MatrixXd K0 = MatrixXd::Zero(this->numberGroups,this->numberGroups);
    
    muint_t pindex = this->numberGroups;
    for(muint_t i=0;i<numberGroups;++i)  {
        K0(i,i) = paramsVC(i);
        for(muint_t j=i+1;j<numberGroups;++j)  {
            K0(i,j) = paramsVC(pindex)*std::sqrt(paramsVC(i)*paramsVC(j));
            K0(j,i) = K0(i,j);
            pindex++;
        }
    }
        
    this->setParamsCovariance(K0);
}
    
void CTFreeFormNew::agetL0(MatrixXd* out) const
{
    /*contruct cholesky factor from hyperparameters*/
    (*out).setConstant(numberGroups,numberGroups,0);
    muint_t pindex=0;
    //for rows
    for(muint_t ir=0;ir<numberGroups;++ir)
        for (muint_t ic=0;ic<(ir+1);++ic)
        {
            (*out)(ir,ic) = params(pindex);
            ++pindex;
        }
}
    
void CTFreeFormNew::agetL0grad_param(MatrixXd* out,muint_t i) const throw(CGPMixException)
{
    /*construct cholesky factors from hyperparameters*/
    (*out).setConstant(numberGroups,numberGroups,0);
    muint_t pindex=0;
    //for rows
    for(muint_t ir=0;ir<numberGroups;++ir)
        for (muint_t ic=0;ic<(ir+1);++ic)
        {
            if (pindex==i)
            {
                (*out)(ir,ic) = 1;
            }
            ++pindex;
        }
}

    
void CTFreeFormNew::agetIparamDiag(VectorXi* out) const
{
    (*out) = VectorXi::Zero(getNumberParams(),1);
    //for rows
    muint_t pindex=0;
    for(muint_t ir=0;ir<numberGroups;++ir)
        for (muint_t ic=0;ic<(ir+1);++ic)
        {
            //diagonal is exponentiated
            if (ic==ir)
                (*out)(pindex) = 1;
            ++pindex;
        }
}
    
    
/* CTDenseNew class */

CTDenseNew::CTDenseNew(muint_t numberGroups) : CTraitNew(numberGroups)
{
    //number of parameters:
    this->numberParams = numberGroups;
}
    
CTDenseNew::~CTDenseNew()
{
}

void CTDenseNew::agetScales(CovarParams* out) {
    (*out) = this->params;
    double sign=1;
    if ((*out)(0)!=0) 	sign = std::abs((*out)(0))/((*out)(0));
    (*out).segment(0,this->numberParams)=sign*(*out).segment(0,this->numberParams);
}

void CTDenseNew::setParamsCovariance(const MatrixXd& K0) throw(CGPMixException)
{
    this->params.resize(this->numberParams);
    //loop over groups
    for(muint_t i=0;i<numberGroups;++i)  {
        params(i) = std::sqrt(K0(i,i));
        params(i)*= K0(i,0)/std::abs(K0(i,0));
    }
}

void CTDenseNew::aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException)
{
    //create template matrix K
    MatrixXd L = this->params;
    (*out).noalias() = L*L.transpose();
}
    
void CTDenseNew::aKgrad_param(MatrixXd* out,muint_t i) const throw(CGPMixException)
{
    MatrixXd L = this->params;
    MatrixXd Lgrad_parami = MatrixXd::Zero(this->numberParams,1);
    Lgrad_parami(i) = 1;
    (*out).noalias() = Lgrad_parami*L.transpose()+L*Lgrad_parami.transpose();
}
    
void CTDenseNew::aKhess_param(MatrixXd* out,muint_t i,muint_t j) const throw(CGPMixException)
{
    MatrixXd Lgrad_parami = MatrixXd::Zero(this->numberParams,1);
    MatrixXd Lgrad_paramj = MatrixXd::Zero(this->numberParams,1);
    Lgrad_parami(i) = 1;
    Lgrad_paramj(j) = 1;
    (*out).noalias() = Lgrad_parami*Lgrad_paramj.transpose()+Lgrad_paramj*Lgrad_parami.transpose();
}

void CTDenseNew::agetParamBounds0(CovarParams* lower,CovarParams* upper) const
{
    *lower = -INFINITY*VectorXd::Ones(this->getNumberParams());
    *upper = +INFINITY*VectorXd::Ones(this->getNumberParams());
}
    
void CTDenseNew::agetParamMask0(CovarParams* out) const
{
    (*out) = VectorXd::Ones(getNumberParams());
}

    
/* CTFixedNew class */

CTFixedNew::CTFixedNew(muint_t numberGroups, const MatrixXd & K0) : CTraitNew(numberGroups)
{
    this->numberParams = 1;
    if(((muint_t)K0.rows()!=numberGroups) || ((muint_t)K0.cols()!=numberGroups))
    {
        throw CGPMixException("Rows and columns need to be compatiable with the number of groups");
    }
    this->K0 = K0;
}
    
CTFixedNew::~CTFixedNew()
{
}

void CTFixedNew::agetScales(CovarParams* out) {
    (*out) = this->params;
    (*out)=(*out).unaryExpr(std::bind2nd( std::ptr_fun<double,double,double>(pow), 2) ).unaryExpr(std::ptr_fun(sqrt));
}
    
void CTFixedNew::setParamsCovariance(const MatrixXd& K0) throw(CGPMixException)
{
    this->params.resize(this->numberParams);
    params(0) = std::sqrt(K0.maxCoeff());
}

void CTFixedNew::aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException)
{
    (*out) = std::pow(params(0),2)*this->K0;
}
    
void CTFixedNew::aKgrad_param(MatrixXd* out,muint_t i) const throw(CGPMixException)
{
    (*out) = 2*params(0)*this->K0;
}

void CTFixedNew::aKhess_param(MatrixXd* out,muint_t i,muint_t j) const throw(CGPMixException)
{
    (*out) = 2*this->K0;
}

void CTFixedNew::agetParamBounds0(CovarParams* lower,CovarParams* upper) const
{
    *lower = -INFINITY*VectorXd::Ones(this->getNumberParams());
    *upper = +INFINITY*VectorXd::Ones(this->getNumberParams());
}

void CTFixedNew::agetParamMask0(CovarParams* out) const
{
    (*out) = VectorXd::Ones(getNumberParams());
}

    
/* CTDiagonalNew class */


CTDiagonalNew::CTDiagonalNew(muint_t numberGroups) : CTraitNew(numberGroups)
{
    this->numberParams = numberGroups;
}
    
CTDiagonalNew::~CTDiagonalNew()
{
}

void CTDiagonalNew::agetScales(CovarParams* out) {
    (*out) = this->params;
    (*out)=(*out).unaryExpr(std::bind2nd( std::ptr_fun<double,double,double>(pow), 2) ).unaryExpr(std::ptr_fun(sqrt));
}
    
void CTDiagonalNew::setParamsCovariance(const MatrixXd& K0) throw(CGPMixException)
{
    this->params.resize(this->numberParams);
    //loop over groups
    for(muint_t i=0;i<numberGroups;++i)  {
        params(i) = std::sqrt(K0(i,i));
    }
}

void CTDiagonalNew::aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException)
{
  (*out) = params.unaryExpr(std::bind2nd( std::ptr_fun<double,double,double>(pow), 2) ).asDiagonal();
}
    
void CTDiagonalNew::aKgrad_param(MatrixXd* out,muint_t i) const throw(CGPMixException)
{
    (*out)=MatrixXd::Zero(numberGroups,numberGroups);
    (*out)(i,i)=2*params(i);
}

void CTDiagonalNew::aKhess_param(MatrixXd* out,muint_t i,muint_t j) const throw(CGPMixException)
{
    (*out)=MatrixXd::Zero(numberGroups,numberGroups);
    if (i==j)   (*out)(i,i)=2;
}

void CTDiagonalNew::agetParamBounds0(CovarParams* lower,CovarParams* upper) const
{
    *lower = -INFINITY*VectorXd::Ones(this->getNumberParams());
    *upper = +INFINITY*VectorXd::Ones(this->getNumberParams());
}
    
void CTDiagonalNew::agetParamMask0(CovarParams* out) const
{
    (*out) = VectorXd::Ones(getNumberParams());
}

    
/* CTLowRankNew class */

CTLowRankNew::CTLowRankNew(muint_t numberGroups) : CTraitNew(numberGroups)
{
    if (numberGroups==2)
        this->numberParams = 3;
    else
        this->numberParams = 2*numberGroups;
}

CTLowRankNew::~CTLowRankNew()
{
}

void CTLowRankNew::agetK0dense(MatrixXd* out) const throw(CGPMixException)
{
    MatrixXd L = this->params.segment(0,this->numberGroups);
    (*out).noalias() = L*L.transpose();
}
    
void CTLowRankNew::agetK0diagonal(MatrixXd* out) const throw(CGPMixException)
{
    if (numberGroups==2)
        (*out) = std::pow(params(2),2)*MatrixXd::Identity(2,2);
    else
      (*out) = params.segment(this->numberGroups,this->numberGroups).unaryExpr(std::bind2nd( std::ptr_fun<double,double,double>(pow), 2) ).asDiagonal();
}
   
void CTLowRankNew::agetScales(CovarParams* out) {
    (*out) = this->params;
    double sign=1;
    if ((*out)(0)!=0) 	sign = std::abs((*out)(0))/((*out)(0));
    if (this->numberGroups==2) {
        (*out)(0)=sign*((*out)(0));
        (*out)(1)=sign*((*out)(1));
        (*out)(2)=std::abs((*out)(2));
    }
    else {
        (*out).segment(0,this->numberGroups)=sign*(*out).segment(0,this->numberGroups);
        (*out).segment(this->numberGroups,this->numberGroups)=(*out).segment(this->numberGroups,this->numberGroups).unaryExpr(std::bind2nd( std::ptr_fun<double,double,double>(pow), 2) ).unaryExpr(std::ptr_fun(sqrt));
    }
}
    
void CTLowRankNew::setParamsCovariance(const MatrixXd& K0) throw(CGPMixException)
{
    //TO DO
}

void CTLowRankNew::aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException)
{
    MatrixXd out1;

    agetK0dense(out);
    agetK0diagonal(&out1);

    (*out)+=out1;
}

void CTLowRankNew::aKgrad_param(MatrixXd* out,muint_t i) const throw(CGPMixException)
{
    if (i>=this->numberParams)
        throw CGPMixException("Index exceeds the number of parameters");

    if (i<this->numberGroups) {
        MatrixXd L = this->params.segment(0,this->numberGroups);
        MatrixXd Lgrad_parami = MatrixXd::Zero(this->numberGroups,1);
        Lgrad_parami(i) = 1;
        (*out).noalias() = Lgrad_parami*L.transpose()+L*Lgrad_parami.transpose();
    }
    else {
        if (numberGroups==2) {
            (*out) = 2*params(2)*MatrixXd::Identity(2,2);
        }
        else {
            (*out)=MatrixXd::Zero(numberGroups,numberGroups);
            (*out)(i-numberGroups,i-numberGroups)=2*params(i);
        }
    }
}

void CTLowRankNew::aKhess_param(MatrixXd* out,muint_t i,muint_t j) const throw(CGPMixException)
{
    if (j>=(muint_t)this->numberParams || j>=(muint_t)this->numberParams)
        throw CGPMixException("Index exceeds the number of parameters");
    
    if (i<this->numberGroups && j<this->numberGroups) {
        MatrixXd Lgrad_parami = MatrixXd::Zero(this->numberGroups,1);
        MatrixXd Lgrad_paramj = MatrixXd::Zero(this->numberGroups,1);
        Lgrad_parami(i) = 1;
        Lgrad_paramj(j) = 1;
        (*out).noalias() = Lgrad_parami*Lgrad_paramj.transpose()+Lgrad_paramj*Lgrad_parami.transpose();
    }
    else if (i>=(muint_t)this->numberGroups && j>=(muint_t)this->numberGroups) {
        if (numberGroups==2) {
            (*out) = 2*MatrixXd::Identity(2,2);
        }
        else {
            (*out)=MatrixXd::Zero(numberGroups,numberGroups);
            if (i==j)   (*out)(i-numberGroups,i-numberGroups)=2;
        }
    }
    else {
        (*out)=MatrixXd::Zero(numberGroups,numberGroups);
    }
}

void CTLowRankNew::agetParamBounds0(CovarParams* lower,CovarParams* upper) const
{

    *lower = -INFINITY*VectorXd::Ones(this->getNumberParams());
    *upper = +INFINITY*VectorXd::Ones(this->getNumberParams());  
}
    
void CTLowRankNew::agetParamMask0(CovarParams* out) const
{
    (*out) = VectorXd::Ones(this->getNumberParams());
}

} /* namespace limix */


