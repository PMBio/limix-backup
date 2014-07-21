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

#include "covariance.h"
#include "limix/utils/matrix_helper.h"

namespace limix {

ACovarianceFunction::ACovarianceFunction(muint_t numberParams)
{
	this->numberParams =numberParams;
	this->params = VectorXd(numberParams);
	//default: 0 dimensions
	this->numberDimensions = 0;
}


ACovarianceFunction::~ACovarianceFunction()
{

}


muint_t ACovarianceFunction::Kdim() const 
{
	if(isnull(X))
		throw CLimixException("ACovarianceFunction: cannot query covariance dimension without X!");
	//standard: use X to determine dimension:
	return X.rows();
}



//set the parameters to a new value.
void ACovarianceFunction::setParams(const CovarParams& params)
{
	checkParamDimensions(params);
	this->params = params;
	//notify child objects that our state has changed
	propagateSync(false);
}

void ACovarianceFunction::agetParams(CovarParams* out) const
{(*out) = params;};


CovarParams ACovarianceFunction::getParams() const
{
	CovarParams rv;
	this->agetParams(&rv);
	return rv;
}


CovarParams ACovarianceFunction::getParamMask() const
{
	CovarParams rv;
	this->agetParamMask(&rv);
	return rv;
}

void ACovarianceFunction::agetParamMask0(CovarParams* out) const {
	(*out) = VectorXd::Ones(getNumberParams());
}


void ACovarianceFunction::agetParamMask(CovarParams* out) const
{
	agetParamMask0(out);
	if(!isnull(paramsMask))
		(*out).array() *= this->paramsMask.array();
}

void ACovarianceFunction::setParamMask(const CovarParams& params)
{
	if((muint_t)params.rows()!=this->getNumberParams())
		throw CLimixException("ACovarianceFunction::setParamMask has illegal shape");
	this->paramsMask = params;
}


void ACovarianceFunction::setNumberDimensions(muint_t numberDimensions)
{
	this->numberDimensions=numberDimensions;
	this->numberParams = 1;
}

muint_t ACovarianceFunction::getNumberParams() const
{return numberParams;}

muint_t ACovarianceFunction::getNumberDimensions() const
{return numberDimensions;}



void ACovarianceFunction::setX(const CovarInput & X) 
{
	checkXDimensions(X);
	this->X = X;
	//notify child objects about sync state change:
	propagateSync(false);
}

void ACovarianceFunction::setXcol(const CovarInput& X,muint_t col) 
{
	//1. check dimensions etc.
	if(((col+X.cols())>(muint_t)this->X.cols()) || (X.rows()!=this->X.rows()))
	{
		std::ostringstream os;
		os << "setXcol out of range. Current X:"<<this->getNumberDimensions() <<")";
		throw CLimixException(os.str());
	}
	this->X.block(0,col,X.rows(),X.cols()) = X;
}

void ACovarianceFunction::agetX(CovarInput *Xout) const 
{
	(*Xout) = this->X;
}



void ACovarianceFunction::aK(MatrixXd* out) const 
{
	aKcross(out,X);
}

void ACovarianceFunction::aKdiag(VectorXd *out) const 
{
	MatrixXd Kfull = K();
	(*out) = Kfull.diagonal();
	return;
}

void ACovarianceFunction::aKgrad_X(MatrixXd *out, const muint_t d) const 
{
	aKcross_grad_X(out,X,d);
}

bool ACovarianceFunction::check_covariance_Kgrad_theta(ACovarianceFunction& covar,mfloat_t relchange,mfloat_t threshold)
{
	mfloat_t RV=0;

	//copy of parameter vector
	CovarParams L = covar.getParams();
	//create copy
	CovarParams L0 = L;
	//dimensions
	for(mint_t i=0;i<L.rows();i++)
	{
		mfloat_t change = relchange*L(i);
		change = std::max(change,1E-5);
		L(i) = L0(i) + change;
		covar.setParams(L);
		MatrixXd Lplus = covar.K();
		L(i) = L0(i) - change;
		covar.setParams(L);
		MatrixXd Lminus = covar.K();
		//numerical gradient
		MatrixXd diff_numerical  = (Lplus-Lminus)/(2.*change);
		//analytical gradient
		covar.setParams(L0);
		MatrixXd diff_analytical = covar.Kgrad_param(i);
		RV += (diff_numerical-diff_analytical).squaredNorm();
	}
	return (RV < threshold);
}

void ACovarianceFunction::agetParamBounds0(CovarParams* lower,
		CovarParams* upper) const
{
	(*lower) = -INFINITY*VectorXd::Ones(getNumberParams());
	(*upper) = +INFINITY*VectorXd::Ones(getNumberParams());
}

void ACovarianceFunction::agetParamBounds(CovarParams* lower,
		CovarParams* upper) const {

	//query intrinsic bound
	agetParamBounds0(lower,upper);

	//override if needed
	if(!isnull(bound_upper))
	{
        for (muint_t i=0; i<this->numberParams; i++)
            if ((this->bound_lower)(i)>(*lower)(i)) (*lower)(i)=(this->bound_lower)(i);
	}
	if(!isnull(bound_lower))
	{
        for (muint_t i=0; i<this->numberParams; i++)
            if ((this->bound_upper)(i)<(*upper)(i)) (*upper)(i)=(this->bound_upper)(i);
	}
}

void ACovarianceFunction::initParams()
{
	VectorXd params = MatrixXd::Zero(this->getNumberParams(),1);
	this->setParams(params);
}


void ACovarianceFunction::setParamBounds(const CovarParams& lower,
		const CovarParams& upper)  {
	if(((muint_t)lower.rows()!=this->numberParams) || ((muint_t)upper.rows()!=this->numberParams)) {
        throw CLimixException("Entry lengths do not coincide with the number of parameters.");
    }
    for (muint_t i=0; i<this->numberParams; i++)
        if (upper(i)<lower(i)) throw CLimixException("Incompatible values.");
	this->bound_lower = lower;
	this->bound_upper = upper;
}

bool ACovarianceFunction::check_covariance_Kgrad_x(ACovarianceFunction& covar,mfloat_t relchange,mfloat_t threshold,bool check_diag)
{
	mfloat_t RV=0;
	//copy inputs for which we calculate gradients
	CovarInput X = covar.getX();
	CovarInput X0 = X;
	for (int ic=0;ic<X.cols();ic++)
	{
		//analytical gradient is per columns all in one go:
		MatrixXd Kgrad_x = covar.Kgrad_X(ic);
		MatrixXd Kgrad_x_diag = covar.Kdiag_grad_X(ic);
		for (int ir=0;ir<X.rows();ir++)
		{
			mfloat_t change = relchange*X0(ir,ic);
			change = std::max(change,1E-5);
			X(ir,ic) = X0(ir,ic) + change;
			covar.setX(X);
			MatrixXd Lplus = covar.K();
			X(ir,ic) = X0(ir,ic) - change;
			covar.setX(X);
			MatrixXd Lminus = covar.K();
			X(ir,ic) = X0(ir,ic);
			covar.setX(X);
			//numerical gradient
			MatrixXd diff_numerical = (Lplus-Lminus)/(2.*change);
			//build analytical gradient matrix
			MatrixXd diff_analytical = MatrixXd::Zero(X.rows(),X.rows());
			diff_analytical.row(ir) = Kgrad_x.row(ir);
			diff_analytical.col(ir) += Kgrad_x.row(ir);
			RV+= (diff_numerical-diff_analytical).squaredNorm();
			//difference
			if (check_diag)
			{
				double delta =(diff_numerical(ir,ir)-Kgrad_x_diag(ir));
				RV+= delta*delta;
			}
		} //end for ir
	}
	return (RV < threshold);
}

void ACovarianceFunction::aKhess_param_num(ACovarianceFunction& covar, MatrixXd* out, const muint_t i, const muint_t j) 
{
    mfloat_t relchange=1E-5;
    
    //copy of parameter vector
    CovarParams L = covar.getParams();
    //create copy
    CovarParams L0 = L;
    //dimensions
    mfloat_t change = relchange*L(j);
    change = std::max(change,1E-5);
    L(j) = L0(j) + change;
    covar.setParams(L);
    covar.aKgrad_param(out,i);
    L(j) = L0(j) - change;
    covar.setParams(L);
    (*out)-=covar.Kgrad_param(i);
    (*out)/=(2.*change);
    covar.setParams(L0);
}

std::ostream& operator<< (std::ostream &out, ACovarianceFunction &cov)
{
	assert(cov.K().rows()>0 && cov.K().cols()>0);

	Eigen::IOFormat fmt(Eigen::StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
	return out << cov.K().format(fmt);
}


//COVAR cache
/*CCovarainceFunctionCache*/
void CCovarianceFunctionCacheOld::setCovar(PCovarianceFunction covar)
	{
		//delete sync child of old covar
		if(this->covar)
			this->covar->delSyncChild(sync);
		//set new covar
		this->covar = covar;
		//register sync handler
		this->covar->addSyncChild(sync);
		//clear cache:
		setSync(false);
	}


void CCovarianceFunctionCacheOld::updateSVD()
{
	Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(rgetK());
	UCache = eigensolver.eigenvectors();
	SCache = eigensolver.eigenvalues();
	SVDCacheNull=false;
}

void CCovarianceFunctionCacheOld::validateCache()
{
	if (!isInSync())
	{
		KCacheNull = true;
		SVDCacheNull = true;
		cholKCacheNull = true;
		//set sync
	}
	setSync();
}
// caching functions
MatrixXd& CCovarianceFunctionCacheOld::rgetK()
{
	validateCache();
	if (KCacheNull)
	{
		covar->aK(&KCache);
		KCacheNull=false;
	}
	return KCache;

}
MatrixXd& CCovarianceFunctionCacheOld::rgetUK()
{
	validateCache();
	if (SVDCacheNull)
		updateSVD();
	return UCache;
}
VectorXd& CCovarianceFunctionCacheOld::rgetSK()
{
	validateCache();
	if (SVDCacheNull)
		updateSVD();
	return SCache;
}

MatrixXdChol& CCovarianceFunctionCacheOld::rgetCholK()
{
	validateCache();
	if(cholKCacheNull)
	{
		cholKCache = MatrixXdChol(rgetK());
		cholKCacheNull=false;
	}
	return cholKCache;
}




}/* namespace limix */


