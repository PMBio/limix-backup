/*
 * CVqtl.cpp
 *
 *  Created on: Jul 26, 2012
 *      Author: stegle
 */

#include "CMultiTrait.h"
#include "limix/utils/matrix_helper.h"
#include "limix/gp/gp_opt.h"
#include "limix/covar/freeform.h"
#include "limix/covar/combinators.h"

namespace limix {





CMultiTraitVQTL::CMultiTraitVQTL(muint_t numtraits) {
	this->numtraits = numtraits;
	// TODO Auto-generated constructor stub

}

CMultiTraitVQTL::~CMultiTraitVQTL() {
	// TODO Auto-generated destructor stub
}


void CMultiTraitVQTL::initGP()
{

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
	this->covar_noise = PFreeFormCF(new CFreeFormCF(this->numtraits));
	this->covar_noise->setX(this->trait);

	//2. construct GP model
	PLikNormalNULL lik(new CLikNormalNULL());
	this->gp = PGPbase(new CGPbase(this->covar,lik));
	//set phenotype
	this->gp->setY(this->pheno);
}

void CMultiTraitVQTL::train()
{
	CGPHyperParams params;
	//CovarInput covar_params =  MatrixXd::Zero();
	MatrixXd covar_params = MatrixXd::Zero(this->covar->getNumberParams(),1);
	params["covar"] = covar_params;

	gp->setParams(params);

	CGPopt opt(gp);
	std::cout << "gradcheck: "<< opt.gradCheck() << "\n";
	opt.opt();
	std::cout << "gradcheck: "<< opt.gradCheck() << "\n";
}


//setters and getters

void CMultiTraitVQTL::agetK(MatrixXd* out,muint_t i) const {
	(*out) = K_terms[i];
}

void CMultiTraitVQTL::agetKgeno(MatrixXd* out) const {
	(*out) = Kgeno;
}

void CMultiTraitVQTL::setK(const MatrixXd& K,muint_t i) {
	this->K_terms[i] = K;
}

void CMultiTraitVQTL::addK(const MatrixXd& K)
{
	this->K_terms.push_back(K);
}


void CMultiTraitVQTL::setKgeno(const MatrixXd& Kgeno) {
	this-> Kgeno = Kgeno;
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

void CMultiTraitVQTL::setTrait(const MatrixXd& trait) {
	this->trait = trait;
}

} /* namespace limix */
