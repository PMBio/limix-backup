// Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
// All rights reserved.
//
// LIMIX is provided under a 2-clause BSD license.
// See license.txt for the complete license.

#include "CGPLMM.h"
#include "limix/utils/gamma.h"

namespace limix {
/*GPLMM*/

CGPLMM::CGPLMM(PGPkronecker gp) : gp(gp)
{
	this->params0 = gp->getParams();
}


void CGPLMM::checkConsistency() 
{
	//check that data term is correct type
	/*
	if (typeid(gp.getDataTerm())!=typeid(CKroneckerMean))
		throw CLimixException("CGPLMM requires a CKroneckerMean data term");
	*/
	//check dimensionality of data structures
	this->num_samples = snps.rows();
	this->num_snps = snps.cols();
	this->num_pheno = pheno.cols();
	this->num_covs = covs.cols();
	if(!num_samples == pheno.rows())
		throw CLimixException("phenotypes and SNP dimensions inconsistent");

	if(!num_samples == covs.rows())
		throw CLimixException("covariates and SNP dimensions inconsistent");

	if (gp)
	{
		//check that SNPs have consisten dimension:
		if(this->num_samples!=gp->getNumberSamples())
			throw CLimixException("GP and kroneckerLMM sample inconsitency");
	}
}




void CGPLMM::initTesting() 
{
	//initialize testing basedon A and A0
	checkConsistency();
	std::cout << "consistent" << "\n";

	MatrixXd fixedEffectsAlt = MatrixXd::Ones(num_samples,1);
	MatrixXd fixedEffects0   = this->covs;

	//0. create pointers for SumLinearMean
	mean0 = PSumLinear(new CSumLinear());
	meanAlt = PSumLinear(new CSumLinear());
	mean0->setY(this->pheno);
	meanAlt->setY(this->pheno);

	//1. loop through terms and create fixed effect object
	//1.1 Null model terms
	for(MatrixXdVec::const_iterator iter = VA0.begin(); iter!=VA0.end();iter++)
	{
		PKroneckerMean term(new CKroneckerMean());
		//set design
		term->setA(iter[0]);
		//set fixed effect
		term->setFixedEffects(fixedEffects0);
		//add this term to both, mean0 and meanAlt
		mean0->appendTerm(term);
		meanAlt->appendTerm(term);
	}
	//1.2 alternative model terms + 1 term for AAlt
	altTerm = PKroneckerMean(new CKroneckerMean());
	//set design
	altTerm->setA(AAlt);
	//set fixed effect
	altTerm->setFixedEffects(fixedEffectsAlt);
	//add this term to both, mean0 and meanAlt
	meanAlt->appendTerm(altTerm);

	//estimate degrees of freedom
	degreesFreedom = fixedEffectsAlt.cols();

	//3. get weight parameters for both dataTerms
	weights0.resize(mean0->getRowsParams(),mean0->getColsParams());
	weightsAlt.resize(meanAlt->getRowsParams(),meanAlt->getColsParams());

	//create hyperparams objects
	hpAlt = params0;
	hp0    = params0;

	//set data Term paramtetes
	hp0["dataTerm"] = weights0;
	hpAlt["dataTerm"] = weightsAlt;

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



PGPkronecker CGPLMM::getGp() const
{
    return gp;
}

void CGPLMM::setGp(PGPkronecker gp)
{
    this->gp = gp;
}


void CGPLMM::process() 
{
	//1. init testing engine
	initTesting();
	//2. init result arrays
	nLL0.resize(1,num_snps);
	nLLAlt.resize(1,num_snps);
	ldelta0.resize(1,num_snps);
	ldeltaAlt.resize(1,num_snps);
	pv.resize(1,num_snps);

	/*
	//initialize covaraites and x
	MatrixXd xAlt  = MatrixXd::Zero(num_samples,1+num_covs);
	MatrixXd x0 = MatrixXd::Zero(num_samples,num_covs);
	xAlt.block(0,1,num_samples,num_covs) = this->covs;
	x0.block(0,0,num_samples,num_covs) = this->covs;
	*/

	//2. loop over SNPs
	mfloat_t deltaNLL;
	for (muint_t is=0;is<num_snps;++is)
	{
		//swap out fixed effect:
		altTerm->setFixedEffects(this->snps.block(0,is,num_samples,1));

		//1. evaluate null model
		gp->setDataTerm(mean0);
		gp->setParams(hp0);
		opt->opt();
		//store NLL
		nLL0(0,is) = gp->LML();
		//store delta
		ldelta0(0,is) = 2*gp->getParams()["lik"](1);

		//2. evaluate alternative model
		gp->setDataTerm(meanAlt);
		gp->setParams(hpAlt);
		opt->opt();
		nLLAlt(0,is) = gp->LML();
		//store delta
		ldeltaAlt(0,is) = 2*gp->getParams()["lik"](1);

		deltaNLL = nLL0(0,is) - nLLAlt(0,is);
		if (deltaNLL<=0)
			deltaNLL = 1E-10;
		//3. pvalues
		this->pv(0, is) = stats::Gamma::gammaQ(deltaNLL, (double)(0.5) * getDegreesFredom());
	}
}

}//end: namespace
