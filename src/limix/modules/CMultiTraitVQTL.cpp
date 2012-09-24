/*
 * CVqtl.cpp
 *
 *  Created on: Jul 26, 2012
 *      Author: stegle
 */

#include "CMultiTraitVQTL.h"
#include "limix/utils/matrix_helper.h"
#include "limix/gp/gp_opt.h"
#include "limix/covar/freeform.h"
#include "limix/covar/se.h"
#include "limix/covar/combinators.h"
#include "limix/mean/CLinearMean.h"
#include "limix/LMM/lmm.h"
namespace limix {





CMultiTraitVQTL::CMultiTraitVQTL()
{

}

CMultiTraitVQTL::~CMultiTraitVQTL() {
	// TODO Auto-generated destructor stub
	estimate_noise_covar = false;
}


void CMultiTraitVQTL::checkConsistency() throw (CGPMixException){
	//check that all data structures supplied are consistent
	//1. determine dimensions from phenotype
	muint_t Nn = this->pheno.rows();
	muint_t Np = this->pheno.cols();
	if (Np!=1)
		throw CGPMixException("CMultiTraitVQTL: phenotypes need to be concatenated, resulting in dimension =1");

	//2. check fixed effects
	if (!isnull(this->fixed))
	{
			if (Nn!=(muint_t)this->fixed.rows())
				throw CGPMixException("CMultiTraitVQTL: fixed effect dimensions inconsistent");
	}
	else
	{
		//create empty fixed effect matrix
		this->fixed = MatrixXd::Zero(Nn,0);
	}

	//3. check covariance matrices
	if (((muint_t)this->Kgeno.rows()!=Nn) || ((muint_t)this->Kgeno.rows()!=Nn))
		throw CGPMixException("CMultiTraitVQTL: Kgeno matrix has inconsistent dimension.");

	for(MatrixXdVec::iterator iter = this->K_terms.begin(); iter!=this->K_terms.end();iter++)
	{
		if (((muint_t)iter->rows()!=Nn) || ((muint_t)iter->cols()!=Nn))
			throw CGPMixException("CMultiTraitVQTL: Kterm matrix has inconsistent dimension.");
	}
}



mfloat_t CMultiTraitVQTL::estimateLogVariance(const MatrixXd& K)
{
	//return the genetic varaince explained by K
	CLMM lmm;
	lmm.setK(K);
	lmm.setSNPs(MatrixXd::Zero(K.rows(),1));
	lmm.setPheno(this->pheno);
	lmm.setCovs(this->fixed);
	lmm.setVarcompApprox0(-20, 10, 1000);
    lmm.process();
    MatrixXd ldelta0 = lmm.getLdelta0();
    MatrixXd lsigma  = lmm.getLSigma();
    return ldelta0(0,0) + lsigma(0,0);
}


PProductCF CMultiTraitVQTL::initCovarTerm(MatrixXd* hp0,
		MatrixXd* hp_mask, const MatrixXd& Kfix, bool categorial_trait, bool trait_covariance)
{
	//0. total term covar is a product
	PProductCF cov_term = PProductCF(new CProductCF());

	//1. fixed CF
	PFixedCF cov_fixed    = PFixedCF(new CFixedCF(Kfix));
	cov_term->addCovariance(cov_fixed);

	//2. trait CF
	PCovarianceFunction cov_trait;
	//2.1 is categorial?
	if(categorial_trait)
	{
		//freeform categorial CF
		cov_trait = PFreeFormCF(new CFreeFormCF(this->numtraits));
	}
	else
	{
		//SE ard covariance
		cov_trait = PCovSqexpARD(new CCovSqexpARD(1));
	}
	cov_term->addCovariance(cov_trait);
	//set input for covar_term
	cov_term->setX(this->trait);

	//initialization of hyperparameters & constraints
	(*hp0) = MatrixXd::Zero(cov_term->getNumberParams(),1);
	//get heritability estimate

	mfloat_t l2var = 0.5 * estimateLogVariance(Kfix);

	//construct param mask
	(*hp_mask) = MatrixXd::Ones(cov_term->getNumberParams(),1);
	//1. exclude scaling factor
	(*hp_mask)(0,0) = 0;
	//2. exclude cross covariances if trait_covariance is false

	if(categorial_trait)
	{
		PFreeFormCF cf = static_pointer_cast<CFreeFormCF>(cov_trait);
		//2. get diagonal element indexb
		MatrixXi Idiag = cf->getIparamDiag();
		for(muint_t i=0;i<(muint_t) Idiag.rows();++i)
		{
			//diagonal has l2var scaling:
			if (Idiag(i,0))
				(*hp0)(i+1,0) = l2var;
			//constrain off-diagonal to be zero?
			if ((!trait_covariance) && (Idiag(i,0)==0))
				(*hp_mask)(i+1,0) = 0;
		}
	}
	else
	{
		//scaling of covariance
		(*hp0)(0+1,0) = l2var;
		//set lengthscale to 0, corresponding to independence of the traits
		(*hp0)(1+1,0) = -10;
		//constrain off-diagonal to be zero?
		if (!trait_covariance)
			(*hp_mask)(1+1,0) = 0;
	}
	return cov_term;
}


void CMultiTraitVQTL::initGP() throw(CGPMixException)
{

	//0. check consistency of kernels
	checkConsistency();

	//1. initialize covariance function
	this->covar = PSumCF(new CSumCF());
	covar_params0.clear();
	covar_params_mask.clear();

	//1.1 loop over covaraince components and create fixed CF
	for(MatrixXdVec::iterator iter = this->K_terms.begin(); iter!=this->K_terms.end();iter++)
	{
		MatrixXd hp0;
		MatrixXd hp_mask;
		PProductCF cov_term = initCovarTerm(&hp0,&hp_mask,iter[0],this->categorial_trait,true);
		this->covar->addCovariance(cov_term);
		this->covar_terms.push_back(cov_term);
		this->covar_params0.push_back(hp0);
		this->covar_params_mask.push_back(hp_mask);
	}
	//1.2 add noise covariance
	PFixedCF    cov_fixed    = PFixedCF(new CFixedCF(this->Kgeno));
	PFreeFormCF cov_freeform = PFreeFormCF(new CFreeFormCF(this->numtraits));
	MatrixXd hp0;
	MatrixXd hp_mask;
	this->covar_noise = initCovarTerm(&hp0,&hp_mask,this->Kgeno,this->categorial_trait,estimate_noise_covar);
	this->covar->addCovariance(this->covar_noise);
	this->covar_params0.push_back(hp0);
	this->covar_params_mask.push_back(hp_mask);
	//std::cout << hp0 << "\n";
	//std::cout << "mask:" << hp_mask << "\n";

	//2. construct GP model
	PLikNormalNULL lik(new CLikNormalNULL());
	//set phenotype and data term
	PLinearMean mean(new CLinearMean(this->pheno,this->fixed));

	this->gp = PGPbase(new CGPbase(this->covar,lik,mean));
	this->gp->setY(this->pheno);


	//2.2 hyperparameters
	CGPHyperParams params;
	//CovarInput covar_params =  MatrixXd::Zero();
	MatrixXd covar_params = MatrixXd::Zero(this->covar->getNumberParams(),1);
	//initialize from covar_params0
	muint_t ip=0;
	for(MatrixXdVec::iterator iter = this->covar_params0.begin(); iter!=this->covar_params0.end();iter++)
	{
		covar_params.block(ip,0,iter[0].rows(),1) = iter[0];
		ip += iter[0].rows();
	}


	//fixed effect weights
	MatrixXd fixed_params = MatrixXd::Zero(fixed.cols(),1);
	params["covar"] = covar_params;
	params["dataTerm"] = fixed_params;

	gp->setParams(params);

	//3. create optimization interface
	this->opt = PGPopt(new CGPopt(gp));
	//3.2 add constraints for the optimization
	MatrixXd covar_mask = MatrixXd::Ones(covar->getNumberParams(),1);
	//a) the fixed covarainces weights can be set to 0
	ip=0;
	for(MatrixXdVec::iterator iter = this->covar_params_mask.begin(); iter!=this->covar_params_mask.end();iter++)
	{
		covar_mask.block(ip,0,iter[0].rows(),1) = iter[0];
		ip += iter[0].rows();
	}
	CGPHyperParams mask;
	mask["covar"] = covar_mask;
	opt->setParamMask(mask);
}


bool CMultiTraitVQTL::train() throw(CGPMixException)
{
	initGP();

	bool rv =  this->opt->opt();
	/*
	std::cout << this->gp->LML() << "\n";
	std::cout << this->gp->LMLgrad() << "\n";

	std::cout << this->gp->getParams() << "\n";

	std::cout << this->covar->getParams() << "\n";
	std::cout << this->covar_terms[0]->getParams() << "\n";
	std::cout << this->covar_noise->getParams() << "\n";

	//std::cout << this->covar->K() << "\n";
	*/
	return rv;
}


//setters and getters

void CMultiTraitVQTL::agetK(MatrixXd* out,muint_t i) const {
	(*out) = K_terms[i];
}

void CMultiTraitVQTL::agetKgeno(MatrixXd* out) const {
	(*out) = Kgeno;
}

void CMultiTraitVQTL::setK(muint_t i,const MatrixXd& K,bool rescale) {
	this->K_terms[i] = K;
}

void CMultiTraitVQTL::addK(const MatrixXd& K,bool rescale)
{
	if (rescale)
	{
		MatrixXd tmp = K;
		scale_K(tmp);
		this->K_terms.push_back(tmp);
	}
	else
		this->K_terms.push_back(K);
}


void CMultiTraitVQTL::setKgeno(const MatrixXd& Kgeno,bool rescale) {
	this-> Kgeno = Kgeno;
	if (rescale)
		scale_K(this->Kgeno);
}

void CMultiTraitVQTL::agetFixed(MatrixXd* out) const {
	(*out) = fixed;
}

void CMultiTraitVQTL::setFixed(const MatrixXd& fixed) {
	this->fixed = fixed;
}

void CMultiTraitVQTL::agetTrait(MatrixXd* out) const {
	(*out) = trait;
}

void CMultiTraitVQTL::setTrait(const MatrixXd& trait,bool categorial)
{
	this->categorial_trait = categorial;
	//1. allocate trait vector
	this->trait = trait;
	this->categorial_trait = categorial;
	if(categorial)
	{
		//2. clear unique set of entries
		mfloat_set utrait;
		//3. add unique things
		for (muint_t it=0;it<(muint_t) trait.rows();++it)
		{
			mfloat_t el = trait(it,0);
			if (utrait.find(el)==utrait.end())
			{
				utrait.insert(el);
			}
		}
		//set number of traits
		this->numtraits=utrait.size();
		//convert to array
		this->utrait = MatrixXd::Zero(numtraits,1);
		muint_t index=0;
		for(mfloat_set::iterator iter = utrait.begin(); iter!=utrait.end();iter++)
		{
			this->utrait(index,0) = iter.operator *();
			index++;
		}
	}//end if categorial
}

void CMultiTraitVQTL::agetPheno(MatrixXd* out) const {
	(*out) = this->pheno;
}


void CMultiTraitVQTL::setPheno(const MatrixXd& pheno) {
	this->pheno = pheno;
}


PCovarianceFunction CMultiTraitVQTL::getCovar_term(muint_t i)
{
		return this->covar_terms[i];
}

PCovarianceFunction CMultiTraitVQTL::getCovar_noise(bool traitCovar)
{

	return this->covar_noise;
}

void  CMultiTraitVQTL::agetFreeFormVariance(MatrixXd* out,CovarParams params)
{
	//1. create freeform covaraince
	CFreeFormCF covar(this->numtraits);
	//2. set unique env factors
	covar.setX(this->utrait);
	//3. set params
	covar.setParams(params);
	//4.evaluate covariance
	covar.aK(out);
}


void CMultiTraitVQTL::agetVarianceComponent_term(MatrixXd* out, muint_t i)
{
	//1. get parameters of covariance term
	CovarParams params;
	this->covar_terms[i]->getCovariance(1)->agetParams(&params);
	//2. evalaute
	this->agetFreeFormVariance(out,params);
}

void CMultiTraitVQTL::agetVarianceComponent_noise(MatrixXd* out)
{


	CGPHyperParams p = gp->getParams();
	gp->setParams(p);
	CovarParams params;
	this->covar_noise->getCovariance(1)->agetParams(&params);
	this->agetFreeFormVariance(out,params);
}

mfloat_t CMultiTraitVQTL::getLML() {
	return this->gp->LML();
}

mfloat_t CMultiTraitVQTL::estimateHeritability(const MatrixXd& Y,
		const MatrixXd& K)
{
	MatrixXd covs = MatrixXd::Ones(Y.rows(),1);
	CLMM lmm;
	lmm.setK(K);
	lmm.setSNPs(MatrixXd::Zero(K.rows(),1));
	lmm.setPheno(Y);
	lmm.setCovs(covs);
	lmm.setVarcompApprox0(-20, 20, 1000);
    lmm.process();
    MatrixXd ldelta0 = lmm.getLdelta0();
    //std::cout << ldelta0 << "\n";
    mfloat_t rv = 1.0/(1.0+exp(ldelta0(0,0)));
    return rv;

}

} /* namespace limix */
