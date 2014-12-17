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

#include "freeform.h"
#include <math.h>
#include "dist.h"
//#ifdef _WIN32
//#include <tgmath.h>
//#endif

namespace limix {

    
/* FREE FORM */
    
    
CFreeFormCF::CFreeFormCF(muint_t numberGroups)
{
    this->numberDimensions = 0;
    this->numberGroups=numberGroups;
    //this->X=MatrixXd::Zero(numberGroups,1);
    //number of parameters:
    this->numberParams = calcNumberParams(numberGroups);
    initParams();
}
    
muint_t CFreeFormCF::calcNumberParams(muint_t numberGroups)
{
    return (0.5*numberGroups*(numberGroups+1));
}
    
CFreeFormCF::~CFreeFormCF()
{
}

void CFreeFormCF::aKcross_diag(VectorXd* out, const CovarInput& Xstar) const 
{
	MatrixXd K;
	aKcross(&K,Xstar);
	(*out)=K.diagonal();
}

void CFreeFormCF::agetScales(CovarParams* out) {
    (*out) = this->params;
    double sign=1;
    if ((*out)(0)!=0) 	sign = std::fabs((*out)(0))/((*out)(0));
    (*out)=sign*(*out);
    (*out)(this->numberParams-1)=std::fabs((*out)(this->numberParams-1));
}
    
void CFreeFormCF::aK0Covar2Params(VectorXd* out,const MatrixXd& K0)
{
}

void CFreeFormCF::setParamsCovariance(const MatrixXd& K0) 
{
    //0. check that the matrix has the correct size
    if(((muint_t)K0.rows()!=numberGroups) || ((muint_t)K0.cols()!=numberGroups))
    {
        throw CLimixException("aK0Covar2Params: rows and columns need to be compatiable with the number of groups");
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

void CFreeFormCF::aKcross(MatrixXd* out, const CovarInput& Xstar ) const  
{
    //create template matrix K
    MatrixXd L;
    agetL0(&L);
    (*out).noalias() = L*L.transpose();
}
    
void CFreeFormCF::aKgrad_param(MatrixXd* out,const muint_t i) const 
{
    MatrixXd L;
    MatrixXd Lgrad_parami;

    agetL0(&L);
    agetL0grad_param(&Lgrad_parami,i);

    //use chain rule K = LL^T
    (*out).noalias() = Lgrad_parami*L.transpose() + L*Lgrad_parami.transpose();
}


void CFreeFormCF::aKhess_param(MatrixXd* out,const muint_t i,const muint_t j) const 
{

    MatrixXd Lgrad_parami;
    MatrixXd Lgrad_paramj;

    agetL0grad_param(&Lgrad_parami,i);
    agetL0grad_param(&Lgrad_paramj,j);

    //use chain rule K = LL^T
    (*out).noalias() = Lgrad_parami*Lgrad_paramj.transpose() + Lgrad_paramj*Lgrad_parami.transpose();
}


void CFreeFormCF::agetParamMask0(CovarParams* out) const {
    (*out) = VectorXd::Ones(getNumberParams());
}
    
    
void CFreeFormCF::setParamsVarCorr(const CovarParams& paramsVC) 
{
    
    // No correlation must be <1 && >-1 or otherwise the initialization would fail
    for (muint_t i=numberGroups;i<getNumberParams();++i)
        if (paramsVC(i)>=1 || paramsVC(i)<=-1)
            throw CLimixException("Correlation must be in (-1,+1)");
    
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
    
void CFreeFormCF::agetL0(MatrixXd* out) const
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
    
void CFreeFormCF::agetL0grad_param(MatrixXd* out,muint_t i) const 
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

    
void CFreeFormCF::agetIparamDiag(VectorXi* out) const
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



/* CRankOneCF class */

CRankOneCF::CRankOneCF(muint_t numberGroups)
{
    this->numberDimensions = 0;
    this->numberGroups=numberGroups;
    //this->X=MatrixXd::Zero(numberGroups,1);
    //number of parameters:
    this->numberParams = numberGroups;
    initParams();
}
    
CRankOneCF::~CRankOneCF()
{
}

void CRankOneCF::aKcross_diag(VectorXd* out, const CovarInput& Xstar) const 
{
	MatrixXd K;
	aKcross(&K,Xstar);
	(*out)=K.diagonal();
}

void CRankOneCF::agetScales(CovarParams* out) {
    (*out) = this->params;
    double sign=1;
    if ((*out)(0)!=0) 	sign = std::fabs((*out)(0))/((*out)(0));
    (*out).segment(0,this->numberParams)=sign*(*out).segment(0,this->numberParams);
}

void CRankOneCF::setParamsCovariance(const MatrixXd& K0) 
{
    this->params.resize(this->numberParams);
    //loop over groups
    for(muint_t i=0;i<numberGroups;++i)  {
        params(i) = std::sqrt(K0(i,i));
        params(i)*= K0(i,0)/std::fabs(K0(i,0));
    }
}

void CRankOneCF::aKcross(MatrixXd* out, const CovarInput& Xstar ) const 
{
    //create template matrix K
    MatrixXd L = this->params;
    (*out).noalias() = L*L.transpose();
}
    
void CRankOneCF::aKgrad_param(MatrixXd* out,muint_t i) const 
{
    MatrixXd L = this->params;
    MatrixXd Lgrad_parami = MatrixXd::Zero(this->numberParams,1);
    Lgrad_parami(i) = 1;
    (*out).noalias() = Lgrad_parami*L.transpose()+L*Lgrad_parami.transpose();
}
    
void CRankOneCF::aKhess_param(MatrixXd* out,muint_t i,muint_t j) const 
{
    MatrixXd Lgrad_parami = MatrixXd::Zero(this->numberParams,1);
    MatrixXd Lgrad_paramj = MatrixXd::Zero(this->numberParams,1);
    Lgrad_parami(i) = 1;
    Lgrad_paramj(j) = 1;
    (*out).noalias() = Lgrad_parami*Lgrad_paramj.transpose()+Lgrad_paramj*Lgrad_parami.transpose();
}
    
void CRankOneCF::agetParamMask0(CovarParams* out) const
{
    (*out) = VectorXd::Ones(getNumberParams());
}


/* CLowRankCF class */

CLowRankCF::CLowRankCF(muint_t numberGroups, muint_t rank)
{
    this->numberDimensions = 0;
    this->numberGroups=numberGroups;
    //this->X=MatrixXd::Zero(numberGroups,1);
    //number of parameters:
    this->numberParams = rank*numberGroups;
    this->rank = rank;
    initParams();
}

CLowRankCF::~CLowRankCF()
{
}

void CLowRankCF::aKcross_diag(VectorXd* out, const CovarInput& Xstar) const 
{
    MatrixXd K;
    aKcross(&K,Xstar);
    (*out)=K.diagonal();
}

void CLowRankCF::agetScales(CovarParams* out) {
    //to implement properly
    (*out) = this->params;
    double sign=1;
    if ((*out)(0)!=0) 	sign = std::fabs((*out)(0))/((*out)(0));
    (*out)*=sign;
}

void CLowRankCF::setParamsCovariance(const MatrixXd& K0) 
{
    //IMPLEMENT ME
    //1. U, S = eigh(K0)
    //2. a_i = U[:,i]*SP.sqrt(S[i])
    //2. params = (a_1,a_2,...)
}

void CLowRankCF::aKcross(MatrixXd* out, const CovarInput& Xstar ) const 
{
    //create template matrix K
    MatrixXd L = this->params;
    L.resize(numberGroups,rank);
    (*out).noalias() = L*L.transpose();
}

void CLowRankCF::aKgrad_param(MatrixXd* out,muint_t i) const 
{
    MatrixXd L = this->params;
    MatrixXd Lgrad_parami = MatrixXd::Zero(this->numberParams,1);
    Lgrad_parami(i) = 1;
    L.resize(numberGroups,rank);
    Lgrad_parami.resize(numberGroups,rank);
    (*out).noalias() = Lgrad_parami*L.transpose()+L*Lgrad_parami.transpose();
}

void CLowRankCF::aKhess_param(MatrixXd* out,muint_t i,muint_t j) const 
{
    MatrixXd Lgrad_parami = MatrixXd::Zero(this->numberParams,1);
    MatrixXd Lgrad_paramj = MatrixXd::Zero(this->numberParams,1);
    Lgrad_parami(i) = 1;
    Lgrad_paramj(j) = 1;
    Lgrad_parami.resize(numberGroups,rank);
    Lgrad_paramj.resize(numberGroups,rank);
    (*out).noalias() = Lgrad_parami*Lgrad_paramj.transpose()+Lgrad_paramj*Lgrad_parami.transpose();
}

void CLowRankCF::agetParamMask0(CovarParams* out) const
{
    (*out) = VectorXd::Ones(getNumberParams());
}

    

/* CFixedCF class */

CFixedCF::CFixedCF(const MatrixXd & K0) : ACovarianceFunction(1)
{
    if((muint_t)K0.rows()!=((muint_t)K0.cols()))
    {
        throw CLimixException("K0 must be a square Matrix");
    }
    this->numberGroups=K0.cols();
    this->numberDimensions = 0;
    this->K0 = K0;
    initParams();
}
    
CFixedCF::~CFixedCF()
{
}

void CFixedCF::agetScales(CovarParams* out) {
    (*out) = this->params;
    (*out)=(*out).unaryExpr(std::bind2nd( std::ptr_fun<double,double,double>(pow), 2) ).unaryExpr(std::ptr_fun(sqrt));
}
    
void CFixedCF::setParamsCovariance(const MatrixXd& K0) 
{
    this->params.resize(this->numberParams);
    params(0) = std::sqrt(K0.maxCoeff());
}

void CFixedCF::aKcross(MatrixXd* out, const CovarInput& Xstar ) const 
{
	(*out) = std::pow(params(0),2)*K0cross;
}

void CFixedCF::aKcross_diag(VectorXd* out, const CovarInput& Xstar) const 
{
	(*out) = std::pow(params(0),2)*K0cross_diag;
}
    
void CFixedCF::aKgrad_param(MatrixXd* out,muint_t i) const 
{
    (*out) = 2*params(0)*this->K0;
}

void CFixedCF::aKhess_param(MatrixXd* out,muint_t i,muint_t j) const 
{
    (*out) = 2*this->K0;
}

void CFixedCF::aKcross_grad_X(MatrixXd *out, const CovarInput & Xstar, const muint_t d) const 
{
	(*out) = MatrixXd::Zero(X.rows(),Xstar.rows());
}

void CFixedCF::aKdiag_grad_X(VectorXd *out, const muint_t d) const 
{
	(*out) = VectorXd::Zero(X.rows());
}

void CFixedCF::aK(MatrixXd* out) const 
{
    (*out) = std::pow(params(0),2)*this->K0;
}

void CFixedCF::agetParamMask0(CovarParams* out) const
{
    (*out) = VectorXd::Ones(getNumberParams());
}

void CFixedCF::setK0(const MatrixXd& K0)
{
	this->K0 = K0;
}

void CFixedCF::setK0cross(const MatrixXd& Kcross)
{
	this->K0cross = Kcross;
}

void CFixedCF::agetK0(MatrixXd *out) const
{
	(*out) = K0;
}

void CFixedCF::agetK0cross(MatrixXd *out) const
{
	(*out) = K0cross;
}

void CFixedCF::setK0cross_diag(const VectorXd& Kcross_diag)
{
	this->K0cross_diag = Kcross_diag;
}

void CFixedCF::agetK0cross_diag(VectorXd *out) const
{
	(*out) = K0cross_diag;
}

    
/* CDiagonalCF class */


CDiagonalCF::CDiagonalCF(muint_t numberGroups)
{
    this->numberGroups=numberGroups;
    this->numberDimensions = 0;
    //this->X=MatrixXd::Zero(numberGroups,1);
    this->numberParams = numberGroups;
    initParams();
}
    
CDiagonalCF::~CDiagonalCF()
{
}

void CDiagonalCF::aKcross_diag(VectorXd* out, const CovarInput& Xstar) const 
{
	MatrixXd K;
	aKcross(&K,Xstar);
	(*out)=K.diagonal();
}

void CDiagonalCF::agetScales(CovarParams* out) {
    (*out) = this->params;
    (*out)=(*out).unaryExpr(std::bind2nd( std::ptr_fun<double,double,double>(pow), 2) ).unaryExpr(std::ptr_fun(sqrt));
}
    
void CDiagonalCF::setParamsCovariance(const MatrixXd& K0) 
{
    this->params.resize(this->numberParams);
    //loop over groups
    for(muint_t i=0;i<numberGroups;++i)  {
        params(i) = std::sqrt(K0(i,i));
    }
}

void CDiagonalCF::aKcross(MatrixXd* out, const CovarInput& Xstar ) const 
{
  (*out) = params.unaryExpr(std::bind2nd( std::ptr_fun<double,double,double>(pow), 2) ).asDiagonal();
}
    
void CDiagonalCF::aKgrad_param(MatrixXd* out,muint_t i) const 
{
    (*out)=MatrixXd::Zero(numberGroups,numberGroups);
    (*out)(i,i)=2*params(i);
}

void CDiagonalCF::aKhess_param(MatrixXd* out,muint_t i,muint_t j) const 
{
    (*out)=MatrixXd::Zero(numberGroups,numberGroups);
    if (i==j)   (*out)(i,i)=2;
}
    
void CDiagonalCF::agetParamMask0(CovarParams* out) const
{
    (*out) = VectorXd::Ones(getNumberParams());
}

    
/* CRank1diagCF class */

CRank1diagCF::CRank1diagCF(muint_t numberGroups)
{
	this->numberGroups = numberGroups;
    this->numberDimensions = 0;
    //this->X=MatrixXd::Zero(numberGroups,1);
    if (numberGroups==2)
        this->numberParams = 3;
    else
        this->numberParams = 2*numberGroups;
    initParams();
}

CRank1diagCF::~CRank1diagCF()
{
}

void CRank1diagCF::aKcross_diag(VectorXd* out, const CovarInput& Xstar) const 
{
	MatrixXd K;
	aKcross(&K,Xstar);
	(*out)=K.diagonal();
}

void CRank1diagCF::agetRank1(MatrixXd* out) const 
{
    MatrixXd L = this->params.segment(0,this->numberGroups);
    (*out).noalias() = L*L.transpose();
}
    
void CRank1diagCF::agetDiag(MatrixXd* out) const 
{
    if (numberGroups==2)
        (*out) = std::pow(params(2),2)*MatrixXd::Identity(2,2);
    else
      (*out) = params.segment(this->numberGroups,this->numberGroups).unaryExpr(std::bind2nd( std::ptr_fun<double,double,double>(pow), 2) ).asDiagonal();
}
   
void CRank1diagCF::agetScales(CovarParams* out) {
    (*out) = this->params;
    double sign=1;
    if ((*out)(0)!=0) 	sign = std::fabs((*out)(0))/((*out)(0));
    if (this->numberGroups==2) {
        (*out)(0)=sign*((*out)(0));
        (*out)(1)=sign*((*out)(1));
        (*out)(2)=std::fabs((*out)(2));
    }
    else {
        (*out).segment(0,this->numberGroups)=sign*(*out).segment(0,this->numberGroups);
        (*out).segment(this->numberGroups,this->numberGroups)=(*out).segment(this->numberGroups,this->numberGroups).unaryExpr(std::bind2nd( std::ptr_fun<double,double,double>(pow), 2) ).unaryExpr(std::ptr_fun(sqrt));
    }
}
    
void CRank1diagCF::setParamsCovariance(const MatrixXd& K0) 
{
    //TO DO
}

void CRank1diagCF::aKcross(MatrixXd* out, const CovarInput& Xstar ) const 
{
    MatrixXd out1;

    agetRank1(out);
    agetDiag(&out1);

    (*out)+=out1;
}

void CRank1diagCF::aKgrad_param(MatrixXd* out,muint_t i) const 
{
    if (i>=this->numberParams)
        throw CLimixException("Index exceeds the number of parameters");

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

void CRank1diagCF::aKhess_param(MatrixXd* out,muint_t i,muint_t j) const 
{
    if (j>=(muint_t)this->numberParams || j>=(muint_t)this->numberParams)
        throw CLimixException("Index exceeds the number of parameters");
    
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
    
void CRank1diagCF::agetParamMask0(CovarParams* out) const
{
    (*out) = VectorXd::Ones(this->getNumberParams());
}


/* CSqExpCF class */

CSqExpCF::CSqExpCF(muint_t numberGroups, muint_t dim)
{
    this->numberDimensions = 0;
    this->numberGroups=numberGroups;
    //this->X=MatrixXd::Zero(numberGroups,1);
    //number of parameters:
    this->numberParams = (dim+1)*numberGroups;
    this->dim = dim;
}

CSqExpCF::~CSqExpCF()
{
}

void CSqExpCF::aKcross_diag(VectorXd* out, const CovarInput& Xstar) const 
{
    MatrixXd K;
    aKcross(&K,Xstar);
    (*out)=K.diagonal();
}

void CSqExpCF::agetScales(CovarParams* out) {
    //to implement properly
    (*out) = this->params;
    double sign=1;
    if ((*out)(0)!=0) 	sign = std::fabs((*out)(0))/((*out)(0));
    (*out)*=sign;
}

void CSqExpCF::setParamsCovariance(const MatrixXd& K0) 
{
    //IMPLEMENT ME
}

void CSqExpCF::aKcross(MatrixXd* out, const CovarInput& Xstar ) const 
{
    //create template matrix K
    MatrixXd X = this->params.block(numberGroups,0,dim*numberGroups,1);
    X.resize(numberGroups,dim);
    MatrixXd l = X.block(0,0,1,dim);
    X.block(0,0,1,dim) = MatrixXd::Ones(1,dim);
    MatrixXd Xl = X * l.asDiagonal();
	//squared exponential distance
	MatrixXd RV;
	sq_dist(&RV,Xl,Xl);
	RV*= -1;
	(*out) = RV.unaryExpr(std::ptr_fun(exp));
	MatrixXd A = this->params.block(0,0,numberGroups,1)*this->params.block(0,0,numberGroups,1).transpose();
	(*out).array()*=A.array();
}

void CSqExpCF::aKgrad_param(MatrixXd* out,muint_t i) const 
{
    //exponential part
    MatrixXd X = this->params.block(numberGroups,0,dim*numberGroups,1);
    X.resize(numberGroups,dim);
    MatrixXd l = X.block(0,0,1,dim);
    X.block(0,0,1,dim) = MatrixXd::Ones(1,dim);
    MatrixXd Xl = X * l.asDiagonal();
	MatrixXd RV;
	sq_dist(&RV,Xl,Xl);
	RV*= -1;
	(*out) = RV.unaryExpr(std::ptr_fun(exp));

	// Derivative
	MatrixXd A;
	if (i>=numberGroups){
		A = this->params.block(0,0,numberGroups,1)*this->params.block(0,0,numberGroups,1).transpose();
		muint_t d = muint_t(i/numberGroups)-1;
		muint_t p = i%numberGroups;
		if (p==0) {
			MatrixXd x = X.block(0,d,numberGroups,1);
			sq_dist(&RV,x,x);
			RV*=-2*l(0,d);
		}
		else {
			RV = MatrixXd::Zero(numberGroups,numberGroups);
			for (muint_t pp=0; pp<numberGroups; pp++) {
				RV(p,pp) = -2*(X(p,d)-X(pp,d))*l(0,d);
				RV(pp,p) = RV(p,pp);
			}
		}
		(*out).array()*=RV.array();
	}
	else {
		MatrixXd a = MatrixXd::Zero(numberGroups,1);
		a(i) = 1;
		A = a*this->params.block(0,0,numberGroups,1).transpose()+this->params.block(0,0,numberGroups,1)*a.transpose();
	}
	(*out).array()*=A.array();
}

void CSqExpCF::aKhess_param(MatrixXd* out,muint_t i,muint_t j) const 
{
    //TODO
	(*out)=MatrixXd::Zero(numberGroups,numberGroups);
}

void CSqExpCF::agetParamMask0(CovarParams* out) const
{
    (*out) = VectorXd::Ones(getNumberParams());
}

/* CFixedDiagonalCF class */

CFixedDiagonalCF::CFixedDiagonalCF(PCovarianceFunction covar, const VectorXd& d)
{
    this->numberGroups=covar->Kdim();
    this->numberDimensions = 0;
    this->covar = covar;
    this->numberParams = covar->getNumberParams();
    this->d = d;
    initParams();
}

CFixedDiagonalCF::~CFixedDiagonalCF()
{
}

void CFixedDiagonalCF::aKcross_diag(VectorXd* out, const CovarInput& Xstar) const
{
    (*out)=this->d;
}

void CFixedDiagonalCF::agetScales(CovarParams* out) {
    (*out) = this->getParams();
}

void CFixedDiagonalCF::setParamsCovariance(const MatrixXd& K0)
{
    //not implemented
}

void CFixedDiagonalCF::aKcross(MatrixXd* out, const CovarInput& Xstar ) const
{
    MatrixXd C = this->covar->K();
    (*out) = MatrixXd::Zero(this->Kdim(),this->Kdim());
    for (muint_t ir=0; ir<(*out).rows(); ir++) {
        for (muint_t ic=0; ic<=ir; ic++) {
            (*out)(ir,ic)=C(ir,ic)*sqrt(d(ir)*d(ic)/(C(ir,ir)*C(ic,ic)));
            (*out)(ic,ir)=(*out)(ir,ic);
        }
    }
}

void CFixedDiagonalCF::aKgrad_param(MatrixXd* out,muint_t i) const
{
    MatrixXd C = this->covar->K();
    MatrixXd Cgrad = this->covar->Kgrad_param(i);
    (*out) = MatrixXd::Zero(this->Kdim(),this->Kdim());
    for (muint_t ir=0; ir<(*out).rows(); ir++) {
        for (muint_t ic=0; ic<=ir; ic++) {
            (*out)(ir,ic)  = Cgrad(ir,ic)/sqrt(C(ir,ir)*C(ic,ic));
            (*out)(ir,ic) += -0.5*C(ir,ic)*Cgrad(ir,ir)/(sqrt(C(ir,ir)*C(ic,ic))*C(ir,ir));
            (*out)(ir,ic) += -0.5*C(ir,ic)*Cgrad(ic,ic)/(sqrt(C(ir,ir)*C(ic,ic))*C(ic,ic));
            (*out)(ir,ic) *= sqrt(d(ir)*d(ic));
            (*out)(ic,ir)=(*out)(ir,ic);
        }
    }
}

void CFixedDiagonalCF::aKhess_param(MatrixXd* out,muint_t i,muint_t j) const
{
    //not implemented
}

void CFixedDiagonalCF::agetParamMask0(CovarParams* out) const
{
    (*out) = VectorXd::Ones(getNumberParams());
}


/* CPolyCF class */

CPolyCF::CPolyCF(muint_t numberGroups, muint_t n_dims, muint_t order)
{
    this->numberDimensions = 0;
    this->numberGroups=numberGroups;
    //number of parameters:
    this->numberParams = order+n_dims*numberGroups;
    this->n_dims = n_dims;
    this->order  = order;
    initParams();
}

CPolyCF::~CPolyCF()
{
}

void CPolyCF::aKcross_diag(VectorXd* out, const CovarInput& Xstar) const 
{
    MatrixXd K;
    aKcross(&K,Xstar);
    (*out)=K.diagonal();
}

void CPolyCF::agetScales(CovarParams* out) {
    //to implement properly
    (*out) = this->params;
}

void CPolyCF::setParamsCovariance(const MatrixXd& K0)
{
    //IMPLEMENT ME
}

void CPolyCF::aKcross(MatrixXd* out, const CovarInput& Xstar ) const 
{
    MatrixXd W = this->params.block(order,0,n_dims*numberGroups,1);
    W.resize(numberGroups,n_dims);
    MatrixXd W2o;
    (*out).setConstant(this->numberGroups,this->numberGroups,0);
    for (muint_t o=1; o<this->order+1; o++) {
        W2o = W.unaryExpr(std::bind2nd(std::ptr_fun<double,double,double>(pow),o));
        (*out) += this->params(o-1)*this->params(o-1)*W2o*W2o.transpose();
    }
}

void CPolyCF::aKgrad_param(MatrixXd* out,muint_t i) const
{
    if (i<this->order) {
        MatrixXd W = this->params.block(order,0,n_dims*numberGroups,1);
        W.resize(numberGroups,n_dims);
        MatrixXd W2o = W.unaryExpr(std::bind2nd(std::ptr_fun<double,double,double>(pow),i+1)); 
        (*out).noalias() = 2*this->params(i)*W2o*W2o.transpose();
    }
    else {
        MatrixXd W = this->params.block(order,0,n_dims*numberGroups,1);
        MatrixXd W2oGrad = MatrixXd::Zero(n_dims*numberGroups,1);
        W.resize(numberGroups,n_dims);
        MatrixXd W2o; //MatrixXd W2oGrad;
        (*out).setConstant(this->numberGroups,this->numberGroups,0);
        for (muint_t o=1; o<this->order+1; o++) {
            W2o = W.unaryExpr(std::bind2nd(std::ptr_fun<double,double,double>(pow),o));
            W2oGrad(i-this->order) = o*std::pow(params(i),o-1);
            W2oGrad.resize(numberGroups,n_dims);
            (*out) += this->params(o-1)*this->params(o-1)*W2oGrad*W2o.transpose();
            (*out) += this->params(o-1)*this->params(o-1)*W2o*W2oGrad.transpose();
            W2oGrad.resize(numberGroups*n_dims,1);
        }
    }
}

void CPolyCF::aKhess_param(MatrixXd* out,muint_t i,muint_t j) const{
    //IMPLEMENT ME
}

void CPolyCF::agetParamMask0(CovarParams* out) const
{
    (*out) = VectorXd::Ones(getNumberParams());
}

} /* namespace limix */

