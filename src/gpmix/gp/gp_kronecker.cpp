/*
 * gp_kronecker.cpp
 *
 *  Created on: Jan 2, 2012
 *      Author: stegle
 */

#include "gp_kronecker.h"
#include "gpmix/utils/matrix_helper.h"

namespace gpmix {


MatrixXd* GPSVDCache::getUK()
{
	if (!isInSync())
		this->clearCache();
	if (isnull(UK))
	{
		Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver((*getK()));
		UK = eigensolver.eigenvectors();
		SK = eigensolver.eigenvalues();
	}
	return &UK;
}

MatrixXd* GPSVDCache::getSK()
{
	if (!isInSync())
			this->clearCache();
	if (isnull(UK))
	{
		Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver((*getK()));
		UK = eigensolver.eigenvectors();
		SK = eigensolver.eigenvalues();
	}
	return &SK;
}

void GPSVDCache::clearCache()
{
	covar.makeSync();

	K = MatrixXd();
	UK =  MatrixXd();
	SK = MatrixXd();
}
bool GPSVDCache::isInSync() const
{
	return covar.isInSync();
}


void CGPKroneckerCache::clearCache()
{
	yrot = MatrixXd();
	Si   = MatrixXd();
	YSi  = MatrixXd();
	Knoise = MatrixXd();
}

bool CGPKroneckerCache::isInSync() const
{
	return gp.cache_r.covar.isInSync() && gp.cache_c.covar.isInSync();
}


CGPkronecker::CGPkronecker(ADataTerm &dataTerm, ACovarianceFunction& covar_r, ACovarianceFunction& covar_c, ALikelihood& lik) : CGPbase(dataTerm,covar_r,lik), covar_r(covar_r), covar_c(covar_c), cache_r((*this),covar_r), cache_c((*this),covar_c)
{


}

CGPkronecker::~CGPkronecker() {
	// TODO Auto-generated destructor stub
}



void CGPkronecker::updateParams() throw (CGPMixException)
		{
	CGPbase::updateParams();
	if (this->params.exists("covar_r"))
		this->covar_r.setParams(this->params["covar_r"]);
	if (this->params.exists("covar_c"))
		this->covar_r.setParams(this->params["covar_c"]);
	if (params.exists("X_r"))
		this->updateX(covar_r,gplvmDimensions_r,params["X_r"]);
	if (params.exists("X_c"))
		this->updateX(covar_c,gplvmDimensions_c,params["X_c"]);
		}


} /* namespace gpmix */
