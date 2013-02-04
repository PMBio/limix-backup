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
}

AVarianceTerm::~AVarianceTerm() {
	// TODO Auto-generated destructor stub
}


void limix::AVarianceTerm::agetK(MatrixXd* out) const {
(*out) = this->K;
}

mfloat_t AVarianceTerm::getVariance() const
{
	MatrixXd K = this->covariance->K();
	return getVarianceK(K);
}

AVarianceTerm::AVarianceTerm(const MatrixXd& K,mfloat_t Vinit) {
	this->K = K;
	if(!isnan(Vinit))
	{
		this->setVarianceInit(Vinit);
	}
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
	//if hyperparams are not initialized, just set variance to 1
	if((muint_t)hp_init.rows()==covariance->getNumberParams())
		covariance->setParams(hp_init);
	else if (!isnull(varianceInit))
		covariance->setParams(VectorXd::Ones(1)*this->varianceInit(0,0));
	else
		covariance->setParams(VectorXd::Zero(1));

	if((muint_t)hp_mask.rows()==covariance->getNumberParams())
		covariance->setParamMask(hp_mask);
	isInitialized = true;
}

/* CCategorialCovarainceTerm*/
CCategorialTraitVarianceTerm::CCategorialTraitVarianceTerm()
{
	//default: include cross Covariance Term
	this->constraint = freeform;
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
	if((muint_t)hp_init.rows()==covariance->getNumberParams())
		this->covariance->setParams(hp_init);
	else
	{
		MatrixXd Vinit;
		if (!isnull(this->varianceInit))
			Vinit = varianceInit;
		else
		{
			Vinit = MatrixXd::Zero(numtraits,numtraits);
			Vinit.diagonal().setConstant(1.0);
		}
		//use Vinit to initialize the matrix
		trait_covariance->setParamsCovariance(Vinit);
		Kcovariance->setParams(VectorXd::Ones(1));
	}
	//have manual mask?
	if((muint_t)hp_mask.rows()==covariance->getNumberParams())
		covariance->setParamMask(hp_mask);
	//else: assume it is all fine but remove the constant covar from opt
	else
	{
		Kcovariance->setParamMask(VectorXd::Zero(1));
	}
	//set constraint type
	trait_covariance->setConstraint(constraint);

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
		const MatrixXd& trait, mfloat_t Vinit,CFreeFromCFConstraitType constraint) : AVarianceTerm(K,Vinit){
	this->setTrait(trait);
	this->constraint = constraint;
}


void CCategorialTraitVarianceTerm::setVarianceInit(mfloat_t vinit)
{
	//1. create matrix with initialization
	MatrixXd CovarInit = MatrixXd::Zero(numtraits,numtraits);
	CovarInit.diagonal().setConstant(vinit);
	this->setCovarianceInit(CovarInit);
}

void CCategorialTraitVarianceTerm::setCovarianceInit(const MatrixXd& K0) throw(CGPMixException)
{
	if (! ( ((muint_t)K0.rows()==numtraits) && ((muint_t)K0.cols()==numtraits)))
	{
		throw CGPMixException("CCategorialTrait: initial trait covariance needs to be N x N");
	}
	this->varianceInit = K0;
}

void CCategorialTraitVarianceTerm::agetCovarianceInit(MatrixXd* out) const
{
	(*out) = varianceInit;
}

void CCategorialTraitVarianceTerm::agetCovarianceFit(MatrixXd* out) const
{
	//1. create freeform covaraince
	CFreeFormCF covar(this->numtraits);
	covar.setConstraint(this->constraint);
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
	initialized = false;
}

CVarianceDecomposition::CVarianceDecomposition(const MatrixXd& pheno,
		const MatrixXd& trait) {
	this->trait =trait;
	this->pheno = pheno;
	initialized = false;
}


void CVarianceDecomposition::initGP() throw(CGPMixException)
		{
	//1. construct covariance function
	covar = PSumCF(new CSumCF());
	for(PVarianceTermVec::iterator iter = this->terms.begin(); iter!=this->terms.end();iter++)
	{
		//initialize
		PVarianceTerm term = iter[0];
		//1. check whether there is an initial variance
		term->initCovariance();
		//add covariance term
		covar->addCovariance(iter[0]->getCovariance());
	}


	//2. initialization of varaince parameters (starting point of optimization)
	//initVariances_simple();


	//3. initialize hyperparameters
	hp_covar0 = covar->getParams();
	hp_covar_mask = covar->getParamMask();

	//3. initialize GP and optimization
	PLikNormalNULL lik(new CLikNormalNULL());
	PLinearMean mean(new CLinearMean(this->pheno,this->fixed));
	gp = PGPbase(new CGPbase(covar,lik,mean));
	gp->setY(pheno);

	MatrixXd fixed_params = MatrixXd::Zero(fixed.cols(),1);

	CGPHyperParams params;
	params["covar"] = hp_covar0;
	params["dataTerm"] = fixed_params;
	gp->setParams(params);
	opt = PGPopt(new CGPopt(gp));
	CGPHyperParams mask;

	mask["covar"] = hp_covar_mask;
	//opt->setParamMask(mask);

	if (false)
	{
		std::cout << params << "\n";
		std::cout << mask << "\n";
	}
	this->setInitialized(true);
}


CVarianceDecomposition::~CVarianceDecomposition() {
}


bool CVarianceDecomposition::train()  throw (CGPMixException)
{
	bool rv = false;

	if (!isInitialized())
		{
			initGP();
		}
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

PVarianceTerm CVarianceDecomposition::getNoise() const
{
	PVarianceTerm rv;
	//iterate over covariances and return the noise covariance
	for(PVarianceTermVec::const_iterator iter = this->terms.begin(); iter!=this->terms.end();iter++)
	{
		/*
		if (iter[0]->isIsNoise())
			rv = iter[0];
			*/
	}
	return rv;
}


void CVarianceDecomposition::addTerm(PVarianceTerm term) {
	terms.push_back(term);
	this->setInitialized(false);
}

void CVarianceDecomposition::initVariances_simple() {
	/*
	 * initialize varaince components based on marginal varinace, ignoring multi trait
	 */

	//varaince for each term ignorign all others:
	VectorXd variancesK = VectorXd::Zero(this->terms.size());
	VectorXd variancesN = VectorXd::Zero(this->terms.size());
	muint_t index = 0;

	//1. loop over all variance components and calculate independent heritability
	for(PVarianceTermVec::iterator iter = this->terms.begin(); iter!=this->terms.end();iter++)
	{
		PVarianceTerm term = iter[0];
		//is term a noise or signal covariance?
		if (term->isIsNoise())
			continue;
		//for signal covariances: estimate varaince components ignoring all others
		VectorXd her;
		aestimateHeritability(&her,this->pheno,this->fixed,term->getK());
		//estimate variance for covariance (K) and noise (N)
		variancesN[index] = her[1];
		variancesK[index] = her[0];
		index++;
	}//end for
	std::cout << variancesK << "\n";
	std::cout << variancesN << "\n";
	//2. sum over marginal variance estimates
	mfloat_t total_variance = variancesK.sum() + variancesN.sum();
	std::cout << "estimated total heritability for initialization: " << total_variance << "\n";

	//TODO: figure out noise covariance:


	//3. loop over varainces again and assign
	index =0;
	for(PVarianceTermVec::iterator iter = this->terms.begin(); iter!=this->terms.end();iter++)
	{
		PVarianceTerm term = iter[0];
		//override initialization for terms without initival varaince settings:
		if (!isnull(term->getHpInit()))
		{
			mfloat_t vinit = variancesK[index]/total_variance;
			term->setVarianceInit(vinit);
		}
		index++;
	}
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
		 bool isNoise, bool fitCrossCovariance,mfloat_t Vinit)
{
	PVarianceTerm term;

	if (type==CVarianceDecomposition::singletrait)
	{
		term = PSingleTraitVarianceTerm(new CSingleTraitVarianceTerm(K,Vinit));
	}
	else if(type==CVarianceDecomposition::categorial)
	{
		CFreeFromCFConstraitType constraint = freeform;
		if(!fitCrossCovariance)
			constraint = diagonal;
		term = PCategorialTraitVarianceTerm(new CCategorialTraitVarianceTerm(K,this->trait,Vinit,constraint));
	}
	else if (type==CVarianceDecomposition::continuous)
	{
		//TODO
	}
	//set noise flag
	term->setIsNoise(isNoise);
	//add term
	this->addTerm(term);
}


} //end:: namespace
