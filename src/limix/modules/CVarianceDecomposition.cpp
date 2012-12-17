/*
 * CVarianceDecomposition.cpp
 *
 *  Created on: Nov 27, 2012
 *      Author: stegle
 */

#include "CVarianceDecomposition.h"
#include "limix/utils/matrix_helper.h"
#include "limix/gp/gp_opt.h"
#include "limix/covar/se.h"
#include "limix/covar/combinators.h"
#include "limix/mean/CLinearMean.h"
#include "limix/LMM/lmm.h"
#include <cmath>

namespace limix {


/* CCategorialCovarainceTerm*/
AVarianceTerm::AVarianceTerm() {
	this->isInitialized = false;
	Vinit= NAN;
}

AVarianceTerm::~AVarianceTerm() {
	// TODO Auto-generated destructor stub
}


void limix::AVarianceTerm::agetK(MatrixXd* out) const {
(*out) = this->K;
}

mfloat_t AVarianceTerm::getVariance() const
{
	return getVarianceK(this->covariance->K());
}

AVarianceTerm::AVarianceTerm(const MatrixXd& K,mfloat_t Vinit) {
	this->K = K;
	this->Vinit = Vinit;
}

void limix::AVarianceTerm::setK(const MatrixXd& K) {
	this->K = K;
}


/* CSingleTraitVarainceTerm*/
CSingleTraitVarianceTerm::CSingleTraitVarianceTerm()
{
}

CSingleTraitVarianceTerm::CSingleTraitVarianceTerm(const MatrixXd& K,mfloat_t Vinit) : AVarianceTerm(K,Vinit)
{
}

CSingleTraitVarianceTerm::~CSingleTraitVarianceTerm() {
}

void CSingleTraitVarianceTerm::initCovariance() throw (CGPMixException)
{
	Kcovariance  = PFixedCF(new CFixedCF(this->K));
	covariance   = Kcovariance;
	if((muint_t)hp0.rows()!=covariance->getNumberParams())
		hp0 = MatrixXd::Zero(covariance->getNumberParams(),1);
	if((muint_t)hp_mask.rows()!=covariance->getNumberParams())
		hp_mask = MatrixXd::Ones(covariance->getNumberParams(),1);
	//standard parameter of overall variance
	mfloat_t l2var;
	if(!std::isnan(Vinit))
		l2var = 0.5 * log(Vinit);
	else
		l2var = 0;
	hp0(0,0) = l2var;
	isInitialized = true;
}

void CSingleTraitVarianceTerm::agetFittedVariance(MatrixXd* out) const {
	(*out) = MatrixXd::Zero(1,1);
	(*out)(0,0) = getVariance();
}



/* CCategorialCovarainceTerm*/
CCategorialTraitVarianceTerm::CCategorialTraitVarianceTerm()
{
	//default: include cross Covariance Term
	this->modelCrossCovariance = true;
}

CCategorialTraitVarianceTerm::~CCategorialTraitVarianceTerm() {
	// TODO Auto-generated destructor stub
}


void limix::CCategorialTraitVarianceTerm::agetTrait(VectorXd* out) const {
	(*out) = this->trait;
}

void CCategorialTraitVarianceTerm::initCovariance() throw (CGPMixException)
{
	//0. check consistency
	if(this->trait.rows()!=this->K.rows())
		throw CGPMixException("CCategorialCovarainceTerm: trait information and K covariance have incompatible shapes");
	//1. total term covar is a product
	covariance = PProductCF(new CProductCF());
	//2. fixed CF
	Kcovariance  = PFixedCF(new CFixedCF(this->K));
	//3. freeform covariance
	trait_covariance = PFreeFormCF(new CFreeFormCF(this->numtraits));
	trait_covariance->setX(this->trait);

	//4. add to overall covariance
	static_pointer_cast<CProductCF>(covariance)->addCovariance(Kcovariance);
	static_pointer_cast<CProductCF>(covariance)->addCovariance(trait_covariance);

	//5. create default parameter settings and constraints for optimizatio procedure
	//initialization of hyperparameters & constraints
	if((muint_t)hp0.rows()!=covariance->getNumberParams())
		hp0 = MatrixXd::Zero(covariance->getNumberParams(),1);
	if((muint_t)hp_mask.rows()!=covariance->getNumberParams())
		hp_mask = MatrixXd::Ones(covariance->getNumberParams(),1);

	//standard parameter of overall variance
	VectorXd l2var = VectorXd::Zero(this->numtraits);
	//loop through initialization if valid
	if(!isnull(this->VinitMarginal))
	{
		if((muint_t)VinitMarginal.rows()!=this->numtraits)
			throw CGPMixException("CCategorialCovarainceTerm::initCovarinace: Variance initializatoin and number of traits incompatible.");
		l2var = this->VinitMarginal;
		logInplace(l2var);
		l2var*=0.5;
	}
	else if(!std::isnan(this->Vinit))
	{
		l2var.setConstant(0.5*log(this->Vinit));
	}
	//fix overall scaling factor
	hp_mask(0,0) = 0;
	//identifiy diagonal elements
	MatrixXi Idiag = trait_covariance->getIparamDiag();
	muint_t i_diag=0;
	for (muint_t i=0;i<(muint_t) Idiag.rows();++i)
	{
		//set scaling prameter on diagonal elements
		if (Idiag(i,0))
		{
			hp0(i+1,0) = l2var(i_diag);
			i_diag+=1;
		}
		else if (!this->modelCrossCovariance)
		{
			//if no diagonal disable optimization unlcess modelCrossCovariance is true
			hp_mask(i+1,0) =0;
		}
	}
	isInitialized = true;
} //::end initialize



void CCategorialTraitVarianceTerm::setTrait(const VectorXd& trait)
{
	this->trait = trait;
	mfloat_set utrait;
	//add unique elements
	for (muint_t it=0;it<(muint_t) trait.rows(); ++it)
	{
		muint_t el = trait(it);
		if(utrait.find(el)==utrait.end())
		{
			utrait.insert(el);
		}
	}
	//number of traits
	this->numtraits = utrait.size();
	//convert to array
	this->utrait = VectorXd::Zero(numtraits);
	muint_t index=0;
	for(mfloat_set::iterator iter = utrait.begin(); iter!=utrait.end();iter++)
	{
		this->utrait(index) = iter.operator *();
		index++;
	}
}

CCategorialTraitVarianceTerm::CCategorialTraitVarianceTerm(const MatrixXd& K,
		const MatrixXd& trait, mfloat_t Vinit,bool fitCrossCovariance) : AVarianceTerm(K,Vinit){
	this->setTrait(trait);
	this->modelCrossCovariance = fitCrossCovariance;
}

void CCategorialTraitVarianceTerm::agetFittedVariance(MatrixXd* out) const
{
	//1. create freeform covaraince
	CFreeFormCF covar(this->numtraits);
	//2. set unique env factors
	covar.setX(this->utrait);
	//3. set params
	CovarParams params = this->trait_covariance->getParams();
	covar.setParams(params);
	//4.evaluate covariance
	covar.aK(out);
}



/* CVarianceDecomposition*/
CVarianceDecomposition::CVarianceDecomposition()
{
}

void CVarianceDecomposition::initGP() throw(CGPMixException) {

	//1. construct covariance function
	covar = PSumCF(new CSumCF());
	for(PVarianceTermVec::iterator iter = this->terms.begin(); iter!=this->terms.end();iter++)
	{
		//initialize
		//TODO: use marginal heritability estimate for initialization
		iter[0]->initCovariance();
		//add covariance term
		covar->addCovariance(iter[0]->getCovariance());
	}
	//2. initialize hyperparameters
	hp_covar0 = MatrixXd::Zero(covar->getNumberParams(),1);
	hp_covar_mask = MatrixXd::Ones(covar->getNumberParams(),1);
	muint_t ip=0;
	for(PVarianceTermVec::iterator iter = this->terms.begin(); iter!=this->terms.end();iter++)
	{
		MatrixXd _hp = iter[0]->getHp0();
		MatrixXd _hp_mask = iter[0]->getHpMask();
		assert(_hp.rows()==_hp_mask.rows());
		hp_covar0.block(ip,0,_hp.rows(),1) = _hp;
		hp_covar_mask.block(ip,0,_hp_mask.rows(),1) = _hp_mask;
		ip+=_hp.rows();
	}
	//3. initialize GP and optimization
	PLikNormalNULL lik(new CLikNormalNULL());
	PLinearMean mean(new CLinearMean(this->pheno,this->fixed));
	gp = PGPbase(new CGPbase(covar,lik,mean));
	gp->setY(pheno);

	MatrixXd fixed_params = MatrixXd::Zero(fixed.cols(),1);

	CGPHyperParams params;
	params["covar"] = hp_covar0;
	params["data_term"] = fixed_params;
	gp->setParams(params);
	opt = PGPopt(new CGPopt(gp));
	CGPHyperParams mask;
	mask["covar"] = hp_covar_mask;
	opt->setParamMask(mask);

	if (true)
	{
		std::cout << params << "\n";
		std::cout << mask << "\n";
	}
}

CVarianceDecomposition::CVarianceDecomposition(const MatrixXd& pheno,
		const MatrixXd& trait) {
	this->trait =trait;
	this->pheno = pheno;
}

CVarianceDecomposition::~CVarianceDecomposition() {
}


bool CVarianceDecomposition::train()  throw (CGPMixException)
{
	bool rv = false;

	initGP();
	rv = this->opt->opt();

	/*
	std::cout << this->covar->K() << "\n";

	std::cout << this->gp->LML() << "\n";
	std::cout << this->gp->LMLgrad() << "\n";

	std::cout << this->gp->getParams() << "\n";

	std::cout << this->covar->getParams() << "\n";
	//std::cout << this->covar->K() << "\n";
	*/

	return rv;
}

PVarianceTerm CVarianceDecomposition::getTerm(muint_t i) const {
	return this->terms[i];
}

void CVarianceDecomposition::addTerm(PVarianceTerm term) {
	terms.push_back(term);

}

void CVarianceDecomposition::aestimateHeritability(VectorXd* out, const MatrixXd& Y, const MatrixXd& fixed, const MatrixXd& K)
{
	/*
	 * estimates the genetic and the noise variance and creates a matrirx object to return them
	 */

	MatrixXd covs;
	if(isnull(fixed))
		covs = MatrixXd::Ones(Y.rows(),1);
	else
		covs = fixed;
	//use mixed model code to estimate heritabiltiy
	CLMM lmm;
	lmm.setK(K);
	lmm.setSNPs(MatrixXd::Zero(K.rows(),1));
	lmm.setPheno(Y);
	lmm.setCovs(covs);
	lmm.setVarcompApprox0(-20, 20, 1000);
    lmm.process();
    mfloat_t delta0 = exp(lmm.getLdelta0()(0,0));
    mfloat_t Vtotal = exp(lmm.getLSigma()(0,0));
    VectorXd rv = VectorXd(2);
    rv(0) = Vtotal;
    rv(1) = Vtotal*delta0;
    (*out) =rv;
}



void CVarianceDecomposition::addTerm(const MatrixXd& K, muint_t type,
		mfloat_t Vinit, bool fitCrossCovariance)
{
	if (type==CVarianceDecomposition::singletrait)
	{
		PVarianceTerm term = PSingleTraitVarianceTerm(new CSingleTraitVarianceTerm(K,Vinit));
	}
	else if(type==CVarianceDecomposition::categorial)
	{
		PVarianceTerm term = PCategorialTraitVarianceTerm(new CCategorialTraitVarianceTerm(K,this->trait,Vinit,fitCrossCovariance));
		this->addTerm(term);
	}
	else if (type==CVarianceDecomposition::continuous)
	{
		//TODO
	}
}

} //end:: namespace
