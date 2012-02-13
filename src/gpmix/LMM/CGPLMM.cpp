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
	MatrixXd fixedEffects = MatrixXd::Ones(num_samples,1+num_covs);
	MatrixXd fixedEffects0 = MatrixXd::Ones(num_samples,num_covs);
	weightsAlt    = 0.5+MatrixXd::Zero(1+num_covs,AAlt.rows()).array();
	weights   = 0.5+MatrixXd::Zero(num_covs,AAlt.rows()).array();

	meanAlt = PKroneckerMean(new CKroneckerMean(this->pheno,weightsAlt,fixedEffects,AAlt));
	mean = PKroneckerMean(new CKroneckerMean(this->pheno,weights,fixedEffects0,A0));

	//create hyperparams objects
	hpAlt = gp->getParams();
	hp    = gp->getParams();
	//set data Term paramtetes
	hpAlt["dataTerm"] = weightsAlt;
	hp["dataTerm"] = weights;
	//init optimization
	opt = PGPopt(new CGPopt(gp));
	//bound hyperparameter Optimization (lik)
	CGPHyperParams upper;
	CGPHyperParams lower;
	upper["lik"] = 5.0*MatrixXd::Ones(1,1);
	lower["lik"] = -5.0*MatrixXd::Ones(1,1);
	opt->setOptBoundLower(lower);
	opt->setOptBoundUpper(upper);



	/*
	//construct mean term for testing
	MatrixXd fixedEffects = MatrixXd::Ones(N,1);
	MatrixXd weights = 0.5+MatrixXd::Zero(1,1).array();
	sptr<CKroneckerMean> data(new CKroneckerMean(y,weights,fixedEffects,A));
	*/
}


MatrixXd CGPLMM::getA() const
{
    return AAlt;
}

MatrixXd CGPLMM::getA0() const
{
    return A0;
}

void CGPLMM::setA0(MatrixXd a0)
{
    A0 = a0;
}

void CGPLMM::setA(MatrixXd a)
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
	//estimate effective degrees of freedom
	//TODO: is this right?
	//count difference of non-zero entries
	muint_t df = this->AAlt.count()-this->A0.count();

	//2. loop over SNPs
	for (muint_t is=0;is<num_snps;++is)
	{
		//1. evaluate null model
		gp->setDataTerm(mean);
		gp->setParams(hp);
		opt->opt();
		nLL0(0,is) = gp->LML();
		//2. evaluate alternative model
		gp->setDataTerm(meanAlt);
		gp->setParams(hpAlt);
		opt->opt();
		nLLAlt(0,is) = gp->LML();
		//3. pvalues
		std::cout << nLL0(0,is) << "--" << nLLAlt(0,is) << "\n";
		this->pv(0, is) = Gamma::gammaQ(nLL0(0, is) - nLLAlt(0, is), (double)(((((((0.5))))))) * df);
	}


}

}//end: namespace
