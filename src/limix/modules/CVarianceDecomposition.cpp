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

#include "CVarianceDecomposition.h"
#include "limix/utils/matrix_helper.h"
#include "limix/mean/CSumLinear.h"
#include "limix/mean/CKroneckerMean.h"
#include "limix/gp/gp_kronSum.h"
#include "limix/LMM/lmm.h"

namespace limix {

/* AVarianceTerm */
AVarianceTerm::AVarianceTerm() {
	this->Knull=true;
	this->fitted=false;
	this->is_init=false;
}

AVarianceTerm::~AVarianceTerm() {
}

muint_t AVarianceTerm::getNumberIndividuals() const 
{
	if (Knull)
		throw CLimixException("CSingleTraitTerm: K needs to be set!");
	return (muint_t)this->K.cols();
}

void AVarianceTerm::setK(const MatrixXd& K) 
{
	if(K.rows()!=K.cols())
		throw CLimixException("AVarianceTerm: K needs to be a squared matrix!");
	this->K=K;
	Kcf = PFixedCF(new CFixedCF(this->K));
	this->Knull = false;
}

void AVarianceTerm::agetK(MatrixXd *out) const 
{
	(*out) = this->K;
}

/* CSingleTraitTerm */

CSingleTraitTerm::CSingleTraitTerm():AVarianceTerm() {
}

CSingleTraitTerm::CSingleTraitTerm(const MatrixXd& K):AVarianceTerm() {
	this->setK(K);
}

void CSingleTraitTerm::setSampleFilter(const MatrixXb& filter) 
{
	throw CLimixException("not implementation error: setSampleFilter");
}


CSingleTraitTerm::~CSingleTraitTerm() {
}

PCovarianceFunction CSingleTraitTerm::getTraitCovar() const 
{
	throw CLimixException("CSingleTraitTerm: Not implemented for SingleTraitTerm");
}

void CSingleTraitTerm::setScales(const VectorXd& scales) 
{
	if (Knull)
		throw CLimixException("CSingleTraitTerm: K needs to be set!");
	this->Kcf->setParams(scales);
}

void CSingleTraitTerm::agetScales(VectorXd* out) const 
{
	if (Knull)
		throw CLimixException("CSingleTraitTerm: K needs to be set!");
	(this->Kcf)->agetParams(out);
}

muint_t CSingleTraitTerm::getNumberScales() const 
{
	if (Knull)
		throw CLimixException("CSingleTraitTerm: K needs to be set!");
	return this->Kcf->getNumberParams();
}

void CSingleTraitTerm::initTerm() 
{
	if (Knull)
		throw CLimixException("CSingleTraitTerm: K needs to be set!");
	this->is_init=true;
}

PCovarianceFunction CSingleTraitTerm::getCovariance() const 
{
	if (!is_init)	throw CLimixException("CSingleTraitTerm: the term is not initialised!");
	return this->Kcf;
}

/* CMultiTraitTerm */

CMultiTraitTerm::CMultiTraitTerm(muint_t P):AVarianceTerm()
{
	this->P=P;
	this->isNull=true;
}

CMultiTraitTerm::CMultiTraitTerm(muint_t P, PCovarianceFunction traitCovar, const MatrixXd& K):AVarianceTerm()
{
	this->P=P;
	this->setTraitCovar(traitCovar);
	this->setK(K);
}

CMultiTraitTerm::~CMultiTraitTerm()
{
}

void CMultiTraitTerm::setTraitCovar(PCovarianceFunction traitCovar) 
{
	this->traitCovariance=traitCovar;
	isNull=false;
}

PCovarianceFunction CMultiTraitTerm::getTraitCovar() const 
{
	return this->traitCovariance;
}

void CMultiTraitTerm::setScales(const VectorXd& scales) 
{
	if (isNull)
		throw CLimixException("CMultiTraitTerm: traitCovariance needs to be set!");
	this->traitCovariance->setParams(scales);
}

void CMultiTraitTerm::agetScales(VectorXd* out) const 
{
	if (isNull)
		throw CLimixException("CMultiTraitTerm: traitCovariance needs to be set!");
	this->traitCovariance->agetParams(out);
}

muint_t CMultiTraitTerm::getNumberScales() const 
{
	if (isNull)
		throw CLimixException("CMultiTraitTerm: traitCovariance needs to be set!");
	return this->traitCovariance->getNumberParams();
}

void CMultiTraitTerm::initTerm() 
{
	if (isNull)		throw CLimixException("CMultiTraitTerm: traitCovariance needs to be set!");
	if (Knull)		throw CLimixException("CMultiTraitTerm: K needs to be set!");
	Kcf->setParams(VectorXd::Ones(1));
	Kcf->setParamMask(VectorXd::Zero(1));
	// InterTrait Covariance Matrix
	covariance = PKroneckerCF(new CKroneckerCF(traitCovariance,Kcf));
	this->is_init=true;
}

void CMultiTraitTerm::setSampleFilter(const MatrixXb& filter) 
{
	if (!is_init)
		throw CLimixException("sample Filter can only be aplied after the term is initialized");
	if (filter.rows()!=this->getNumberIndividuals()*this->P)
		throw CLimixException("filter dimensions do not match sample covariance");

	//linearize filter
	MatrixXb filter_ = filter;
	filter_.resize(filter.rows()*filter.cols(),1);
	//get full kronecker index and subset
	MatrixXi kroneckerindex;
	CKroneckerCF::createKroneckerIndex(&kroneckerindex,this->P,this->getNumberIndividuals());
	//subset
	MatrixXi kroneckerindex_;
	//std::cout << kroneckerindex;
	//std::cout << "\n" << "-----------------" << "\n";
	//kroneckerindex_.resize(kroneckerindex.rows(),kroneckerindex.cols());
	slice(kroneckerindex,filter_,kroneckerindex_);
	//std::cout << kroneckerindex_;
	//std::cout << "\n" << "-----------------" << "\n";
	//set as Kroneckerindex
	this->covariance->setKroneckerIndicator(kroneckerindex_);
}


PCovarianceFunction CMultiTraitTerm::getCovariance() const 
{
	if (!is_init)	throw CLimixException("CMultiTraitTerm: the term is not initialised!");
	return this->covariance;
}


/* CVarianceDecomposition */
CVarianceDecomposition::CVarianceDecomposition(const MatrixXd& pheno){
	this->setPheno(pheno);
    this->is_init=false;
    this->fast=false;
}

CVarianceDecomposition::~CVarianceDecomposition(){
}

void CVarianceDecomposition::clear()
{
	this->fixedEffs.clear();
	this->designs.clear();
	this->terms.clear();
	this->is_init=false;
}

void CVarianceDecomposition::addFixedEffTerm(const MatrixXd& design, const MatrixXd& fixed) 
{
	//if ((muint_t)fixed.cols()!=(muint_t)1 || (muint_t)fixed.rows()!=this->N)
	if ((muint_t)fixed.cols()<(muint_t)1 || (muint_t)fixed.rows()!=this->N)
		throw CLimixException("CVarianceDecomposition: the fixed effect must have shape (N,1+)");
	if ((muint_t)design.cols()!=(muint_t)P || (muint_t)design.rows()>(muint_t)P)
		throw CLimixException("CVarianceDecomposition: the design must have P columns and cannot have more than P rows");
	fixedEffs.push_back(fixed);
	designs.push_back(design);
	this->is_init=false;
}

void CVarianceDecomposition::addFixedEffTerm(const MatrixXd& fixed) 
{
	MatrixXd design = MatrixXd::Identity(P,P);
	addFixedEffTerm(fixed,design);
}

void CVarianceDecomposition::getFixed(MatrixXd *out, const muint_t i) const 
{
	if (i>=this->getNumberFixedEffs())
		throw CLimixException("CVarianceDecomposition: value out of range");
	(*out)=this->fixedEffs[i];
}

void CVarianceDecomposition::getDesign(MatrixXd *out, const muint_t i) const 
{
	if (i>=this->getNumberFixedEffs())
		throw CLimixException("CVarianceDecomposition: value out of range");
	(*out)=this->designs[i];
}

void CVarianceDecomposition::clearFixedEffs()
{
	this->fixedEffs.clear();
	this->designs.clear();
	this->is_init=false;
}

muint_t CVarianceDecomposition::getNumberFixedEffs() const
{
	return (muint_t)(this->fixedEffs.size());
}

void CVarianceDecomposition::setPheno(const MatrixXd& pheno) 
{
	// Set Phenoa and dimensions
	this->pheno = pheno;
	this->N = (muint_t)pheno.rows();
	this->P = (muint_t)pheno.cols();
	//check whether phenotype has NANs?
	phenoNAN = isnan(this->pheno);
	this->phenoNANany = phenoNAN.any();
}

void CVarianceDecomposition::getPheno(MatrixXd *out) const 
{
	(*out)=this->pheno;
}

void CVarianceDecomposition::addTerm(PVarianceTerm term) 
{

	if (term->getName()=="CMultiTraitTerm")
		if (term->getNumberTraits()!=this->P)
			throw CLimixException("CVarianceDecomposition: the term has incompatible number of traits");
		if (term->getNumberIndividuals()!=this->N)
			throw CLimixException("CVarianceDecomposition: the term has incompatible number of individual");
	else if (term->getName()=="CSingleTraitTerm")
		if (term->getNumberIndividuals()!=this->N*this->P)
			throw CLimixException("CVarianceDecomposition: the single trait term must have dimensions NP");
	terms.push_back(term);
	this->is_init=false;
}

void CVarianceDecomposition::addTerm(const MatrixXd& K) 
{
//TODO
}

void CVarianceDecomposition::addTerm(PCovarianceFunction traitCovar, const MatrixXd& K) 
{
	this->addTerm(PMultiTraitTerm(new CMultiTraitTerm(traitCovar->Kdim(),traitCovar,K)));
}

PVarianceTerm CVarianceDecomposition::getTerm(muint_t i) const 
{
	if (i>=this->getNumberTerms())
		throw CLimixException("CVarianceDecomposition: value out of range");
	return this->terms[i];
}

void CVarianceDecomposition::clearTerms()
{
	this->terms.clear();
}

muint_t CVarianceDecomposition::getNumberTerms() const
{
	return (muint_t)(terms.size());
}

void CVarianceDecomposition::setScales(const VectorXd& scales) const 
{
	if (this->is_init==0)
		throw CLimixException("CVarianceDecomposition: CVarianceDecomposition needs to be initialised");
	this->covar->setParams(scales);
}

void CVarianceDecomposition::setScales(muint_t i,const VectorXd& scales) const 
{
	if (i>=this->getNumberTerms())
		throw CLimixException("CVarianceDecomposition: value out of range");
	this->terms[i]->setScales(scales);
}

void CVarianceDecomposition::agetScales(muint_t i, VectorXd* out) const 
{
	if (i>=this->getNumberTerms())
		throw CLimixException("CVarianceDecomposition: value out of range");
	this->terms[i]->agetScales(out);
}

void CVarianceDecomposition::agetScales(VectorXd* out) 
{
	(*out).resize(this->getNumberScales(),1);
	muint_t row=0;
	for(PVarianceTermVec::iterator iter = this->terms.begin(); iter!=this->terms.end();iter++)
	{
		PVarianceTerm term = iter[0];
		VectorXd scales;
		term->agetScales(&scales);
		(*out).block(row,0,term->getNumberScales(),1)=scales;
		row+=term->getNumberScales();
	}
}

muint_t CVarianceDecomposition::getNumberScales() 
{
	muint_t out=0;
	for(PVarianceTermVec::iterator iter = this->terms.begin(); iter!=this->terms.end();iter++)
	{
		PVarianceTerm term = iter[0];
		out+=term->getNumberScales();
	}
	return out;
}


void CVarianceDecomposition::initGP(bool fast) 
{
	if (fast)	initGPkronSum();
	else		initGPbase();
}

void CVarianceDecomposition::initGPparams() 
{
	/* get params from covariance matrices and set them to the GP object
 	*/
    if (is_init!=1)
		throw CLimixException("CVarianceDecomposition:: initGP before initGPparams");
    CGPHyperParams params;
	if (fast) {
        params["covarr1"] = static_pointer_cast<CGPkronSum>(gp)->getCovarr1()->getParams();
        params["covarc1"] = static_pointer_cast<CGPkronSum>(gp)->getCovarc1()->getParams();
        params["covarr2"] = static_pointer_cast<CGPkronSum>(gp)->getCovarr2()->getParams();
        params["covarc2"] = static_pointer_cast<CGPkronSum>(gp)->getCovarc2()->getParams();
        params["dataTerm"] = gp->getDataTerm()->getParams();
        gp->setParams(params);
	}
	else {
    	params["covar"] = gp->getCovar()->getParams();
		muint_t ncols = static_pointer_cast<CLinearMean>(gp->getDataTerm())->getRowsParams();
    	params["dataTerm"] = MatrixXd::Zero(ncols,1);
    	gp->setParams(params);
	}
}

void CVarianceDecomposition::initGPbase() 
{
	this->covar = PSumCF(new CSumCF());
	// Init Covariances and sum them
	for(PVarianceTermVec::iterator iter = this->terms.begin(); iter!=this->terms.end();iter++)
	{
		PVarianceTerm term = iter[0];
		term->initTerm();
		this->covar->addCovariance(iter[0]->getCovariance());
	}
	//Build Fixed Effect Term
	// count number of cols
	muint_t numberCols = 0;
	MatrixXdVec::const_iterator design_iter = this->designs.begin();
	MatrixXdVec::const_iterator fixed_iter = this->fixedEffs.begin();
	for (; design_iter != this->designs.end(); design_iter++, fixed_iter++)
		numberCols += design_iter[0].rows()*fixed_iter[0].cols();
	//define fixed for CLinearMean
	MatrixXd fixed(this->N*this->P,numberCols);
	design_iter = this->designs.begin();
	fixed_iter = this->fixedEffs.begin();
	muint_t ncols = 0;
	for (; design_iter != this->designs.end(); design_iter++, fixed_iter++)
	{
		MatrixXd part;
		akron(part,design_iter[0].transpose(),fixed_iter[0]);
		//fixed.block(0,ncols,this->N*this->P,design_iter[0].rows())=part;//Christoph: bug? should also take into account the number of columns in the fixed effects
		fixed.block(0,ncols,this->N*this->P,design_iter[0].rows()*fixed_iter[0].cols())=part;//Christoph: fix
		ncols+=design_iter[0].rows()*fixed_iter[0].cols();
	}

	//vectorize phenotype
	MatrixXd y = this->pheno;
	y.resize(this->N*this->P,1);

	//do we have to deal with missing values
	//phenoNANany = true;
	if(this->phenoNANany)
	{
		//1. vectorize missing values
		MatrixXb Iselect = this->phenoNAN;
		Iselect.resize(this->N*this->P,1);
		//invert
		Iselect = Iselect.unaryExpr(std::ptr_fun(negate));
		//2. select on y
		MatrixXd _y;
		slice(y,Iselect,_y);
		y  = _y;
		//3 fixed effecfs
		MatrixXd _fixed;
		slice(fixed,Iselect,_fixed);
		fixed = _fixed;
		//4. set filter in terms
		for(PVarianceTermVec::iterator iter = this->terms.begin(); iter!=this->terms.end();iter++)
		{
			PVarianceTerm term = iter[0];
			term->setSampleFilter(Iselect);
		}
	}

	//Define Likelihood, LinearMean and GP
	PLikNormalNULL lik(new CLikNormalNULL());
	PLinearMean mean(new CLinearMean(y,fixed));
	this->gp = PGPbase(new CGPbase(covar, lik, mean));
	this->gp->setY(y);
	this->fast=false;
	this->is_init=1;
	//Initialize Params
	this->initGPparams();
	//optimizer
	this->opt = PGPopt(new CGPopt(gp));
}

void CVarianceDecomposition::initGPkronSum() 
{
	//check whether exact Kronecker structure?
	if (this->phenoNANany)
			throw CLimixException("GPKronSum (fast inference) can only be used for full kronecker structured data");

    if (this->getNumberTerms()!=2)
        throw CLimixException("CVarianceDecomposition: fastGP only works for two terms");
    if (this->getNumberTraits()<2)
        throw CLimixException("CVarianceDecomposition: supported only for multiple traits");

    if (this->is_init && this->fast) {
        this->gp->setY(pheno);
        CGPHyperParams params = this->gp->getParams();
        VectorXd covarParams;
        this->agetScales(0,&covarParams); params["covarc1"]=covarParams;
        this->agetScales(1,&covarParams); params["covarc2"]=covarParams;
        params["dataTerm"] = MatrixXd::Zero(params["dataTerm"].rows(),params["dataTerm"].cols());
        this->gp->setParams(params);
    }
    else
    {
        //init covars
        this->terms[0]->initTerm();
        this->terms[1]->initTerm();
		PCovarianceFunction covarr1 = this->terms[0]->getKcf();
		PCovarianceFunction covarr2 = this->terms[1]->getKcf();
		PCovarianceFunction covarc1 = this->terms[0]->getTraitCovar();
		PCovarianceFunction covarc2 = this->terms[1]->getTraitCovar();
        //init dataTerm
		MatrixXdVec::const_iterator fixed_iter = this->fixedEffs.begin();
        MatrixXdVec::const_iterator design_iter  = this->designs.begin();
        PSumLinear mean(new CSumLinear());
		for (; design_iter != this->designs.end(); design_iter++, fixed_iter++) {
            MatrixXd A = design_iter[0];
            MatrixXd F = fixed_iter[0];
            MatrixXd W = MatrixXd::Zero(F.cols(),A.rows());
            mean->appendTerm(PKroneckerMean(new CKroneckerMean(pheno,W,F,A)));
        }
        // init lik
        PLikNormalNULL lik(new CLikNormalNULL());
        //define gpKronSum
		this->gp = PGPkronSum(new CGPkronSum(pheno, covarr1, covarc1, covarr2, covarc2, lik, mean));
        this->fast=true;
        this->is_init=1;
        //Initialize Params
		this->initGPparams();
		//Optimizer GP
        this->opt = PGPopt(new CGPopt(gp));
    }
}


bool CVarianceDecomposition::trainGP() 
{

	bool conv = false;
	// initGP if is not init
	if (this->is_init==0)	this->initGP();

	//train GP
	conv = this->opt->opt();

	//check convergence
    VectorXd scales;
    this->agetScales(&scales);
    conv &= (scales.unaryExpr(std::bind2nd( std::ptr_fun<double,double,double>(pow), 2) ).maxCoeff()<(mfloat_t)10.0);

	return conv;
}

void CVarianceDecomposition::getFixedEffects(VectorXd* out) 
{
	(*out)=this->gp->getParams()["dataTerm"];
}

mfloat_t CVarianceDecomposition::getLML() 
{
	if (!this->is_init)
		throw CLimixException("CVarianceDecomposition: the term is not initialised!");
	return -1.*this->gp->LML();
}

mfloat_t CVarianceDecomposition::getLMLgrad() 
{
	if (!this->is_init)
		throw CLimixException("CVarianceDecomposition: the term is not initialised!");
	float out;
	if (this->fast) 	out = getLMLgradGPkronSum();
	else				out = getLMLgradGPbase();
	return out;
}

mfloat_t CVarianceDecomposition::getLMLgradGPbase() 
{
	if (!this->is_init)
		throw CLimixException("CVarianceDecomposition: the term is not initialised!");
	mfloat_t out = 0;
	// Squared Norm of LMLgrad["covar"]
	VectorXd grad = this->gp->LMLgrad()["covar"];
	VectorXd filter = this->gp->getParamMask()["covar"];
	for (muint_t i=0; i<(muint_t)grad.rows(); i++)
		if (filter(i)==1)	out +=std::pow(grad(i),2);
	// Squared Norm of LMLgrad["dataTerm"]
	grad = this->gp->LMLgrad()["dataTerm"];
	for (muint_t i=0; i<(muint_t)grad.rows(); i++)	out +=std::pow(grad(i),2);
	// Square Root
	out = std::sqrt(out);
	return out;
}

mfloat_t CVarianceDecomposition::getLMLgradGPkronSum() 
{
	mfloat_t out = 0;

	VectorXd grad = this->gp->LMLgrad()["covarc1"];
	for (muint_t i=0; i<(muint_t)grad.rows(); i++)	out +=std::pow(grad(i),2);
	grad = this->gp->LMLgrad()["covarc2"];
		for (muint_t i=0; i<(muint_t)grad.rows(); i++)	out +=std::pow(grad(i),2);
	// Square Root
	out = std::sqrt(out);

	return out;
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



} //end:: namespace
