/*
 * freeform.cpp
 *
 *  Created on: Jan 16, 2012
 *      Author: stegle
 */

#include "freeform.h"
#include <math.h>
#include <cmath>


namespace limix {

CFreeFormCF::CFreeFormCF(muint_t numberGroups,CFreeFromCFConstraitType constraint)
{
	//1 input dimension which selects the group:
	this->numberDimensions = 1;
	//number of groups and parameters:
	this->numberGroups= numberGroups;
	this->numberParams = calcNumberParams(numberGroups);
	this->constraint = constraint;
}

muint_t CFreeFormCF::calcNumberParams(muint_t numberGroups)
{
	return (0.5*numberGroups*(numberGroups-1) + numberGroups);
}

CFreeFormCF::~CFreeFormCF()
{
}

void CFreeFormCF::aK0Covar2Params(VectorXd* out,const MatrixXd& K0)
{
}


void CFreeFormCF::setParamsCovariance(const MatrixXd& K0) throw(CGPMixException)
{
	//0. check that the matrix has the correct size
	if(((muint_t)K0.rows()!=numberGroups) || ((muint_t)K0.cols()!=numberGroups))
	{
		throw CGPMixException("aK0Covar2Params: rows and columns need to be compatiable with the number of groups");
	}
	MatrixXd L;
	//1. is this a constraint freeofrm (diagonal matrix or dense?)
	//if yes, forrce K0 to be diagonal.
	if((constraint!=freeform))
	{
		MatrixXd _K0 = MatrixXd::Zero(K0.rows(),K0.cols());
		_K0.diagonal() = K0.diagonal();
		MatrixXdChol chol(_K0);
		L = chol.matrixL();
	}
	else
	{
		MatrixXdChol chol(K0);
		L =chol.matrixL();
	}
	//2. create output argument and fill
	this->params.resize(calcNumberParams(numberGroups));
	//3. loop over groups
	muint_t pindex=0;
	for(muint_t ir=0;ir<numberGroups;++ir)
		for (muint_t ic=0;ic<(ir+1);++ic)
		{
			params(pindex) = L(ir,ic);
			//constraint?
			//set offdiagonal parameter values to 0
			++pindex;
		}
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

void CFreeFormCF::agetL0_dense(MatrixXd* out) const
{
	(*out).resize(numberGroups,1);
	//L = vector of diagonal entries of covariance
	muint_t pindex=0;
	muint_t lindex=0;
	//for rows
	for(muint_t ir=0;ir<numberGroups;++ir)
		for (muint_t ic=0;ic<(ir+1);++ic)
		{
			if(ir==ic)
			{
				(*out)(lindex) = params(pindex);
				++lindex;
			}
			++pindex;
		}
}


void CFreeFormCF::agetL0grad_param(MatrixXd* out,muint_t i) const throw(CGPMixException)
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


void CFreeFormCF::agetL0grad_param_dense(MatrixXd* out, muint_t i) const throw(CGPMixException)
{
	(*out).setConstant(numberGroups,1,0);
	muint_t pindex=0;
	muint_t lindex=0;
	//for rows
	for(muint_t ir=0;ir<numberGroups;++ir)
		for (muint_t ic=0;ic<(ir+1);++ic)
		{
			mfloat_t deriv =0;
			if (pindex==i)
				deriv = 1;
			if (ir==ic)
			{
				(*out)(lindex) = deriv;
				++lindex;
			}
			++pindex;
		}
}

void CFreeFormCF::agetParamBounds0(CovarParams* lower,CovarParams* upper) const
{
	//all parameters but the diagonal elements are unbounded:
	*lower = -INFINITY*VectorXd::Ones(this->getNumberParams());
	*upper = +INFINITY*VectorXd::Ones(this->getNumberParams());
	//get diagonal elements
	VectorXi isDiagonal =getIparamDiag();
	//set diagonal elements to be bounded [0,inf]
	for (muint_t i=0;i<getNumberParams();++i)
	{
		if(isDiagonal(i))
		{
			(*lower)(i) = 0;
		}
	}
}


void CFreeFormCF::agetK0(MatrixXd* out) const
{
	//create template matrix K
	MatrixXd L;
	if(constraint==dense)
		agetL0_dense(&L);
	else
		agetL0(&L);
	(*out).noalias() = L*L.transpose();
}



void CFreeFormCF::agetK0grad_param(MatrixXd* out,muint_t i) const throw(CGPMixException)
{
	MatrixXd L;
	MatrixXd Lgrad_parami;
	if(constraint==dense)
	{
		agetL0_dense(&L);
		agetL0grad_param_dense(&Lgrad_parami,i);
	}
	else
	{
		agetL0(&L);
		agetL0grad_param(&Lgrad_parami,i);
	}
	//use chain rule K = LL^T
	(*out).noalias() = Lgrad_parami*L.transpose() + L*Lgrad_parami.transpose();
}

void CFreeFormCF::projectKcross(MatrixXd* out,const MatrixXd& K0,const CovarInput& Xstar) const throw (CGPMixException)
{
	//2. loop thourugh entries and create effective covariance:
	//index for param
	(*out).setConstant(Xstar.rows(),X.rows(),0.0);
	MatrixXd Km_rc;
	VectorXd Xb;
	VectorXd Xstarb;
	for (muint_t ir=0;ir<numberGroups;++ir)
		for(muint_t ic=0;ic<numberGroups;++ic)
		{
			//get relevant data entries (N,N) binary matrix
			Xstarb = (Xstar.array()==(mfloat_t)ir).cast<mfloat_t>();
			Xb 	   = (X.array()==(mfloat_t)ic).cast<mfloat_t>();
			//add component to K
			Km_rc.noalias() = K0(ir,ic)* (Xstarb*Xb.transpose());
			(*out) += Km_rc;
		}
}

void CFreeFormCF::aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException)
{
	checkXDimensions(Xstar);
	//1. get K0 Matrix, the template for all others
	MatrixXd K0;
	agetK0(&K0);
	projectKcross(out,K0,Xstar);
}

void CFreeFormCF::aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException)
{
	checkXDimensions(Xstar);
	//dignal matrix of Kstar
	//1. get K0 Matrix, the template for all others
	MatrixXd KK;
	agetK0(&KK);
	(*out).resize(Xstar.rows());
	for (muint_t ir=0;ir<(muint_t)Xstar.rows();++ir)
	{
		//get index
		muint_t index = Xstar(ir,0);
		//set element
		(*out)(ir) = KK(index,index);
	}
}

void CFreeFormCF::aKgrad_param(MatrixXd* out,const muint_t i) const throw(CGPMixException)
{
	checkWithinParams(i);
	// same as Kcross, however using a different base matrix K0
	//1. get K0 Matrix, the template for all others
	MatrixXd K0;
	agetK0grad_param(&K0,i);
	projectKcross(out,K0,X);
}
    
void CFreeFormCF::aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const throw(CGPMixException)
{
    throw CGPMixException("Not implemented yet.");
}

void CFreeFormCF::aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const throw(CGPMixException)
{
}

void CFreeFormCF::aKdiag_grad_X(VectorXd* out,const muint_t d) const throw(CGPMixException)
{
}

void CFreeFormCF::agetParamMask0(CovarParams* out) const {
	//default: no mask
	(*out) = VectorXd::Ones(getNumberParams());
	//cases for different constraints
	if((constraint==diagonal) || (constraint==dense))
	{
		muint_t pindex=0;
		for(muint_t ir=0;ir<numberGroups;++ir)
			for (muint_t ic=0;ic<(ir+1);++ic)
			{
				//off diagonal entry is masked out
				if (ic!=ir)
					(*out)(pindex) = 0;
				++pindex;
			}
	} //end if constraint == diagonal or constraint==dense
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
    
    
/* CTraitCF */
    
    
CTraitCF::CTraitCF(muint_t numberGroups)
{
    //1 input dimension which selects the group:
    this->numberDimensions = 1;
    //number of groups:
    this->numberGroups= numberGroups;
}
    
CTraitCF::~CTraitCF()
{
}
    
muint_t CTraitCF::getNumberGroups() const
{return numberGroups;}
    
void CTraitCF::projectKcross(MatrixXd* out,const MatrixXd& K0,const CovarInput& Xstar) const throw (CGPMixException)
{
    //2. loop thourugh entries and create effective covariance:
    //index for param
    (*out).setConstant(Xstar.rows(),X.rows(),0.0);
    MatrixXd Km_rc;
    VectorXd Xb;
    VectorXd Xstarb;
    for (muint_t ir=0;ir<numberGroups;++ir)
        for(muint_t ic=0;ic<numberGroups;++ic)
        {
            //get relevant data entries (N,N) binary matrix
            Xstarb = (Xstar.array()==(mfloat_t)ir).cast<mfloat_t>();
            Xb 	   = (X.array()==(mfloat_t)ic).cast<mfloat_t>();
            //add component to K
            Km_rc.noalias() = K0(ir,ic)* (Xstarb*Xb.transpose());
            (*out) += Km_rc;
        }
}
    
void CTraitCF::aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException)
{
    checkXDimensions(Xstar);
    //1. get K0 Matrix, the template for all others
    MatrixXd K0;
    agetK0(&K0);
    projectKcross(out,K0,Xstar);
}
    
void CTraitCF::aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException)
{
    checkXDimensions(Xstar);
    //dignal matrix of Kstar
    //1. get K0 Matrix, the template for all others
    MatrixXd KK;
    agetK0(&KK);
    (*out).resize(Xstar.rows());
    for (muint_t ir=0;ir<(muint_t)Xstar.rows();++ir)
    {
        //get index
        muint_t index = Xstar(ir,0);
        //set element
        (*out)(ir) = KK(index,index);
    }
}
    
void CTraitCF::aKgrad_param(MatrixXd* out,const muint_t i) const throw(CGPMixException)
{
    checkWithinParams(i);
    // same as Kcross, however using a different base matrix K0
    //1. get K0 Matrix, the template for all others
    MatrixXd K0;
    agetK0grad_param(&K0,i);
    projectKcross(out,K0,X);
}
    
void CTraitCF::aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const throw(CGPMixException)
{
    checkWithinParams(i);
    checkWithinParams(j);
    // same as Kcross, however using a different base matrix K0
    //1. get K0 Matrix, the template for all others
    MatrixXd D2K0;
    agetK0hess_param(&D2K0,i,j);
    projectKcross(out,D2K0,X);
}
    
void CTraitCF::aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const throw(CGPMixException)
{
}
    
void CTraitCF::aKdiag_grad_X(VectorXd* out,const muint_t d) const throw(CGPMixException)
{
}
    
    
/* FREE FORM */
    
    
CTFreeFormCF::CTFreeFormCF(muint_t numberGroups) : CTraitCF(numberGroups)
{
    //number of parameters:
    this->numberParams = calcNumberParams(numberGroups);
}
    
muint_t CTFreeFormCF::calcNumberParams(muint_t numberGroups)
{
    return (0.5*numberGroups*(numberGroups+1));
}
    
CTFreeFormCF::~CTFreeFormCF()
{
}
    
void CTFreeFormCF::aK0Covar2Params(VectorXd* out,const MatrixXd& K0)
{
}
    
    
void CTFreeFormCF::agetK0(MatrixXd* out) const  throw(CGPMixException)
{
    //create template matrix K
    MatrixXd L;
    agetL0(&L);
    (*out).noalias() = L*L.transpose();
}
    
    
    
void CTFreeFormCF::agetK0grad_param(MatrixXd* out,muint_t i) const throw(CGPMixException)
{
    MatrixXd L;
    MatrixXd Lgrad_parami;
    
    agetL0(&L);
    agetL0grad_param(&Lgrad_parami,i);
    
    //use chain rule K = LL^T
    (*out).noalias() = Lgrad_parami*L.transpose() + L*Lgrad_parami.transpose();
}

    
void CTFreeFormCF::agetK0hess_param(MatrixXd* out,muint_t i,muint_t j) const throw(CGPMixException)
{

    MatrixXd Lgrad_parami;
    MatrixXd Lgrad_paramj;

    agetL0grad_param(&Lgrad_parami,i);
    agetL0grad_param(&Lgrad_paramj,j);
        
    //use chain rule K = LL^T
    (*out).noalias() = Lgrad_parami*Lgrad_paramj.transpose() + Lgrad_paramj*Lgrad_parami.transpose();
}
    
    
    
void CTFreeFormCF::setParamsCovariance(const MatrixXd& K0) throw(CGPMixException)
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
    
    
void CTFreeFormCF::agetParamBounds0(CovarParams* lower,CovarParams* upper) const
{
    //all parameters but the diagonal elements are unbounded:
    *lower = -INFINITY*VectorXd::Ones(this->getNumberParams());
    *upper = +INFINITY*VectorXd::Ones(this->getNumberParams());
    //get diagonal elements
    VectorXi isDiagonal =getIparamDiag();
    //set diagonal elements to be bounded [0,inf]
    for (muint_t i=0;i<getNumberParams();++i)
    {
        if(isDiagonal(i))
        {
            (*lower)(i) = 0;
        }
    }
}
    
void CTFreeFormCF::agetParamMask0(CovarParams* out) const {
    (*out) = VectorXd::Ones(getNumberParams());
}
    
    
void CTFreeFormCF::setParamsVarCorr(const CovarParams& paramsVC) throw(CGPMixException)
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
    
void CTFreeFormCF::agetL0(MatrixXd* out) const
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
    
void CTFreeFormCF::agetL0grad_param(MatrixXd* out,muint_t i) const throw(CGPMixException)
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

    
void CTFreeFormCF::agetIparamDiag(VectorXi* out) const
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
    
    
/* CTDenseCF class */
    
CTDenseCF::CTDenseCF(muint_t numberGroups) : CTraitCF(numberGroups)
{
    //number of parameters:
    this->numberParams = numberGroups;
}
    
CTDenseCF::~CTDenseCF()
{
}
    
void CTDenseCF::agetK0(MatrixXd* out) const throw(CGPMixException)
{
    //create template matrix K
    MatrixXd L = this->params;
    (*out).noalias() = L*L.transpose();
}
    
void CTDenseCF::agetK0grad_param(MatrixXd* out,muint_t i) const throw(CGPMixException)
{
    MatrixXd L = this->params;
    MatrixXd Lgrad_parami = MatrixXd::Zero(this->numberParams,1);
    Lgrad_parami(i) = 1;
    (*out).noalias() = Lgrad_parami*L.transpose()+L*Lgrad_parami.transpose();
}
    
void CTDenseCF::agetK0hess_param(MatrixXd* out,muint_t i,muint_t j) const throw(CGPMixException)
{
    MatrixXd Lgrad_parami = MatrixXd::Zero(this->numberParams,1);
    MatrixXd Lgrad_paramj = MatrixXd::Zero(this->numberParams,1);
    Lgrad_parami(i) = 1;
    Lgrad_paramj(j) = 1;
    (*out).noalias() = Lgrad_parami*Lgrad_paramj.transpose()+Lgrad_paramj*Lgrad_parami.transpose();
}
    
void CTDenseCF::setParamsCovariance(const MatrixXd& K0) throw(CGPMixException)
{
    this->params.resize(this->numberParams);
    //loop over groups
    for(muint_t i=0;i<numberGroups;++i)  {
        params(i) = std::sqrt(K0(i,i));
        params(i)*= K0(i,0)/std::abs(K0(i,0));
    }
}
    
void CTDenseCF::agetParamBounds0(CovarParams* lower,CovarParams* upper) const
{
    *lower = -INFINITY*VectorXd::Ones(this->getNumberParams());
    *upper = +INFINITY*VectorXd::Ones(this->getNumberParams());
    //set a1 to be 0
    (*lower)(0) = 0;
}
    
void CTDenseCF::agetParamMask0(CovarParams* out) const
{
    (*out) = VectorXd::Ones(getNumberParams());
}
    
    
/* CTFixedCF class */

CTFixedCF::CTFixedCF(muint_t numberGroups, const MatrixXd & K0) : CTraitCF(numberGroups)
{
    this->numberParams = 1;
    if(((muint_t)K0.rows()!=numberGroups) || ((muint_t)K0.cols()!=numberGroups))
    {
        throw CGPMixException("Rows and columns need to be compatiable with the number of groups");
    }
    this->K0 = K0;
}
    
CTFixedCF::~CTFixedCF()
{
}
    
void CTFixedCF::agetK0(MatrixXd* out) const throw(CGPMixException)
{
    (*out) = std::pow(params(0),2)*this->K0;
}
    
void CTFixedCF::agetK0grad_param(MatrixXd* out,muint_t i) const throw(CGPMixException)
{
    (*out) = 2*params(0)*this->K0;
}
    
void CTFixedCF::agetK0hess_param(MatrixXd* out,muint_t i,muint_t j) const throw(CGPMixException)
{
    (*out) = 2*this->K0;
}
    
void CTFixedCF::setParamsCovariance(const MatrixXd& K0) throw(CGPMixException)
{
    this->params.resize(this->numberParams);
    params(0) = std::sqrt(K0.maxCoeff());
}
    
void CTFixedCF::agetParamBounds0(CovarParams* lower,CovarParams* upper) const
{
    *lower = VectorXd::Zero(this->getNumberParams());
    *upper = +INFINITY*VectorXd::Ones(this->getNumberParams());
}

void CTFixedCF::agetParamMask0(CovarParams* out) const
{
    (*out) = VectorXd::Ones(getNumberParams());
}
    
    
/* CTDiagonalCF class */
    
CTDiagonalCF::CTDiagonalCF(muint_t numberGroups) : CTraitCF(numberGroups)
{
    this->numberParams = numberGroups;
}
    
CTDiagonalCF::~CTDiagonalCF()
{
}

void CTDiagonalCF::agetScales(CovarParams* out) {
    (*out) = this->params;
    (*out)=(*out).unaryExpr(std::bind2nd( std::ptr_fun(pow), 2) ).unaryExpr(std::ptr_fun(sqrt));
}


void CTDiagonalCF::agetK0(MatrixXd* out) const throw(CGPMixException)
{
    (*out) = params.unaryExpr(std::bind2nd( std::ptr_fun(pow), 2) ).asDiagonal();
}
    
void CTDiagonalCF::agetK0grad_param(MatrixXd* out,muint_t i) const throw(CGPMixException)
{
    (*out)=MatrixXd::Zero(numberGroups,numberGroups);
    (*out)(i,i)=2*params(i);
}
    
void CTDiagonalCF::agetK0hess_param(MatrixXd* out,muint_t i,muint_t j) const throw(CGPMixException)
{
    (*out)=MatrixXd::Zero(numberGroups,numberGroups);
    if (i==j)   (*out)(i,i)=2;
}
    
void CTDiagonalCF::setParamsCovariance(const MatrixXd& K0) throw(CGPMixException)
{
    this->params.resize(this->numberParams);
    //loop over groups
    for(muint_t i=0;i<numberGroups;++i)  {
        params(i) = std::sqrt(K0(i,i));
    }
}
    
void CTDiagonalCF::agetParamBounds0(CovarParams* lower,CovarParams* upper) const
{
    *lower = -INFINITY*VectorXd::Ones(this->getNumberParams());
    *upper = +INFINITY*VectorXd::Ones(this->getNumberParams());
}
    
void CTDiagonalCF::agetParamMask0(CovarParams* out) const
{
    (*out) = VectorXd::Ones(getNumberParams());
}

    
    
/* CTLowRankCF class */
    
CTLowRankCF::CTLowRankCF(muint_t numberGroups) : CTraitCF(numberGroups)
{
    if (numberGroups==2)
        this->numberParams = 3;
    else
        this->numberParams = 2*numberGroups;
}
    
CTLowRankCF::~CTLowRankCF()
{
}

void CTLowRankCF::agetK0dense(MatrixXd* out) const throw(CGPMixException)
{
    MatrixXd L = this->params.segment(0,this->numberGroups);
    (*out).noalias() = L*L.transpose();
}
    
void CTLowRankCF::agetK0diagonal(MatrixXd* out) const throw(CGPMixException)
{
    if (numberGroups==2)
        (*out) = std::pow(params(2),2)*MatrixXd::Identity(2,2);
    else
        (*out) = params.segment(this->numberGroups,this->numberGroups).unaryExpr(std::bind2nd( std::ptr_fun(pow), 2) ).asDiagonal();
}
   
void CTLowRankCF::agetScales(CovarParams* out) {
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
        (*out).segment(this->numberGroups,this->numberGroups)=(*out).segment(this->numberGroups,this->numberGroups).unaryExpr(std::bind2nd( std::ptr_fun(pow), 2) ).unaryExpr(std::ptr_fun(sqrt));
    }
}


void CTLowRankCF::agetK0(MatrixXd* out) const throw(CGPMixException)
{
    MatrixXd out1;
    
    agetK0dense(out);
    agetK0diagonal(&out1);
    
    (*out)+=out1;
}
    
void CTLowRankCF::agetK0grad_param(MatrixXd* out,muint_t i) const throw(CGPMixException)
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

void CTLowRankCF::agetK0hess_param(MatrixXd* out,muint_t i,muint_t j) const throw(CGPMixException)
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
    
void CTLowRankCF::setParamsCovariance(const MatrixXd& K0) throw(CGPMixException)
{
    //TO DO
}
    
void CTLowRankCF::agetParamBounds0(CovarParams* lower,CovarParams* upper) const
{

    *lower = -INFINITY*VectorXd::Ones(this->getNumberParams());
    *upper = +INFINITY*VectorXd::Ones(this->getNumberParams());  
	
    /* OLD IMPLEMENTATION
    *lower = VectorXd::Zero(this->getNumberParams());
    *upper = +INFINITY*VectorXd::Ones(this->getNumberParams());
    
    (*lower).segment(1,(this->numberGroups)-1) = -INFINITY*VectorXd::Ones((this->numberGroups)-1);
    */
}
    
void CTLowRankCF::agetParamMask0(CovarParams* out) const
{
    (*out) = VectorXd::Ones(this->getNumberParams());
}


} /* namespace limix */


