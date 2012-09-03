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
#include "limix/covar/combinators.h"
#include "limix/mean/CLinearMean.h"

namespace limix {





CMultiTraitVQTL::CMultiTraitVQTL()
{

}

CMultiTraitVQTL::~CMultiTraitVQTL() {
	// TODO Auto-generated destructor stub
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


void CMultiTraitVQTL::initGP() throw(CGPMixException)
{

	//0. check consistency of kernels
	checkConsistency();

	//1. initialize covariance function
	this->covar = PSumCF(new CSumCF());

	//1.1 loop over covaraince components and create fixed CF
	for(MatrixXdVec::iterator iter = this->K_terms.begin(); iter!=this->K_terms.end();iter++)
	{
		MatrixXd Kfix = iter[0];
		PFixedCF cov_fixed    = PFixedCF(new CFixedCF(Kfix));
		PFreeFormCF cov_freeform = PFreeFormCF(new CFreeFormCF(this->numtraits));
		PProductCF cov_term = PProductCF(new CProductCF());
		cov_term->addCovariance(cov_fixed);
		cov_term->addCovariance(cov_freeform);
		this->covar->addCovariance(cov_term);
		this->covar_terms.push_back(cov_term);
		//set input for covar_term
		cov_term->setX(this->trait);
	}
	//1.2 add noise covariance
	PFixedCF    cov_fixed    = PFixedCF(new CFixedCF(this->Kgeno));
	PFreeFormCF cov_freeform = PFreeFormCF(new CFreeFormCF(this->numtraits));
	this->covar_noise = PProductCF(new CProductCF());
	this->covar_noise->addCovariance(cov_fixed);
	this->covar_noise->addCovariance(cov_freeform);
	this->covar_noise->setX(this->trait);
	this->covar->addCovariance(this->covar_noise);


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
	//fixed effect weights
	MatrixXd fixed_params = MatrixXd::Zero(fixed.cols(),1);
	params["covar"] = covar_params;
	params["dataTerm"] = fixed_params;

	gp->setParams(params);

	//3. create optimization interface
	this->opt = PGPopt(new CGPopt(gp));


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

void CMultiTraitVQTL::setK(const MatrixXd& K,muint_t i,bool rescale) {
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

void CMultiTraitVQTL::setTrait(const MatrixXd& trait)
{
	//1. allocate trait vector
	this->trait = trait;
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

} /* namespace limix */
