/*
 * CGPLMM.cpp
 *
 *  Created on: Feb 13, 2012
 *      Author: stegle
 */

#include "CGPLMM.h"
#include "gpmix/utils/gamma.h"

namespace gpmix {
/*GPLMM*/

CGPLMM::CGPLMM(PGPkronecker gp) : gp(gp)
{
	this->params0 = gp->getParams();
}


void CGPLMM::checkConsistency() throw (CGPMixException)
{
	//check that data term is correct type
	/*
	if (typeid(gp.getDataTerm())!=typeid(CKroneckerMean))
		throw CGPMixException("CGPLMM requires a CKroneckerMean data term");
	*/
	//check dimensionality of data structures
	this->num_samples = snps.rows();
	this->num_snps = snps.cols();
	this->num_pheno = pheno.cols();
	this->num_covs = covs.cols();
	if(!num_samples == pheno.rows())
		throw new CGPMixException("phenotypes and SNP dimensions inconsistent");

	if(!num_samples == covs.rows())
		throw CGPMixException("covariates and SNP dimensions inconsistent");

	//check that SNPs have consisten dimension:
	if(this->num_samples!=gp->getNumberSamples())
		throw CGPMixException("GP and kroneckerLMM sample inconsitency");
}

void CGPLMM::initTesting() throw (CGPMixException)
{
	//initialize testing basedon A and A0
	checkConsistency();
	std::cout << "consistent" << "\n";
	MatrixXd fixedEffects = MatrixXd::Ones(num_samples,1+num_covs);
	MatrixXd fixedEffects0 = MatrixXd::Ones(num_samples,num_covs);
	weightsAlt    = 0.5+MatrixXd::Zero(1+num_covs,AAlt.rows()).array();
	weights0   = 0.5+MatrixXd::Zero(num_covs,A0.rows()).array();

	meanAlt = PKroneckerMean(new CKroneckerMean(this->pheno,weightsAlt,fixedEffects,AAlt));
	mean0 = PKroneckerMean(new CKroneckerMean(this->pheno,weights0,fixedEffects0,A0));

	//create hyperparams objects
	hpAlt = params0;
	hp0    = params0;
	//set data Term paramtetes
	hpAlt["dataTerm"] = weightsAlt;
	hp0["dataTerm"] = weights0;
	//init optimization
	opt = PGPopt(new CGPopt(gp));
	//set filter
	opt->setParamMask(paramsMask);
	//bound all parameters
	CGPHyperParams upper;
	CGPHyperParams lower;
	for(CGPHyperParams::const_iterator iter = params0.begin(); iter!=params0.end();iter++)
	{
		MatrixXd value = (*iter).second;
		std::string name = (*iter).first;
		MatrixXd bound_l = -10.0*MatrixXd::Ones(value.rows(),value.cols());
		MatrixXd bound_u = +10.0*MatrixXd::Ones(value.rows(),value.cols());
		lower[name] = bound_l;
		upper[name] = bound_u;
	}
	opt->setOptBoundLower(lower);
	opt->setOptBoundUpper(upper);
}


MatrixXd CGPLMM::getA() const
{
    return AAlt;
}

MatrixXd CGPLMM::getA0() const
{
    return A0;
}

void CGPLMM::setA0(const MatrixXd& a0)
{
    A0 = a0;
}

void CGPLMM::setA(const MatrixXd& a)
{
    AAlt = a;
}

void CGPLMM::agetA0(MatrixXd* out) const
{
	(*out) = A0;
}

PGPkronecker CGPLMM::getGp() const
{
    return gp;
}

void CGPLMM::setGp(PGPkronecker gp)
{
    this->gp = gp;
}

void CGPLMM::agetA(MatrixXd *out) const
{
    (*out) = AAlt;
}

void CGPLMM::process() throw (CGPMixException)
{
	//1. init testing engine
	initTesting();
	//2. init result arrays
	nLL0.resize(1,num_snps);
	nLLAlt.resize(1,num_snps);
	pv.resize(1,num_snps);

	//estimate effective degrees of freedom: differnece in number of weights
	muint_t df = this->AAlt.rows()-this->A0.rows();


	//initialize covaraites and x
	MatrixXd xAlt  = MatrixXd::Zero(num_samples,1+num_covs);
	MatrixXd x0 = MatrixXd::Zero(num_samples,num_covs);
	xAlt.block(0,1,num_samples,num_covs) = this->covs;
	x0.block(0,0,num_samples,num_covs) = this->covs;


	//2. loop over SNPs
	mfloat_t deltaNLL;
	for (muint_t is=0;is<num_snps;++is)
	{
		//0. update mean term
		xAlt.block(0,0,num_samples,1) = this->snps.block(0,is,num_samples,1);
		meanAlt->setFixedEffects(xAlt);

		//1. evaluate null model
		gp->setDataTerm(mean0);
		gp->setParams(hp0);
		opt->opt();	//TODO crashes as number params to optimize is 3, while boundaries is 3
		nLL0(0,is) = gp->LML();
		//2. evaluate alternative model
		gp->setDataTerm(meanAlt);
		gp->setParams(hpAlt);
		opt->opt();
		nLLAlt(0,is) = gp->LML();
		deltaNLL = nLL0(0,is) - nLLAlt(0,is);
		if (deltaNLL<=0)
			deltaNLL = 1E-10;
		//3. pvalues
		this->pv(0, is) = Gamma::gammaQ(deltaNLL, (double)(0.5) * df);
	}


}

}//end: namespace
