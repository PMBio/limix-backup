/*
 * CVarianceDecomposition.cpp
 *
 *  Created on: Nov 27, 2012
 *      Author: stegle
 */

#include "CVarianceDecomposition.h"
#include "limix/utils/matrix_helper.h"
#include <ctime>

namespace limix {

/* AVarianceTerm */
AVarianceTerm::AVarianceTerm() {
	this->K_mode=-1;
	this->fitted=(bool)0;
	this->is_init=(bool)0;
}

AVarianceTerm::~AVarianceTerm() {
}

muint_t AVarianceTerm::getNumberIndividuals() const throw(CGPMixException)
{
	if (K_mode==-1)
		throw CGPMixException("CSingleTraitTerm: K needs to be set!");
	return (muint_t)this->K.cols();
}

void AVarianceTerm::setK(const MatrixXd& K) throw(CGPMixException)
{
	if(K.rows()!=K.cols())
		throw CGPMixException("AVarianceTerm: K needs to be a squared matrix!");
	this->K = K/K.diagonal().mean();
	Kcf = PTFixed(new CTFixed(K.rows(),this->K));
	this->K_mode = 0;
}

void AVarianceTerm::setX(const MatrixXd& X) throw(CGPMixException)
{
	this->X = X;
	MatrixXd K = X*X.transpose();
	this->setK(K);
	this->K_mode = 1;
}

void AVarianceTerm::agetK(MatrixXd *out) const throw(CGPMixException)
{
	(*out) = this->K;
}

void AVarianceTerm::agetX(MatrixXd *out) const throw(CGPMixException)
{
	(*out) = this->X;
}

/* CSingleTraitTerm */

CSingleTraitTerm::CSingleTraitTerm():AVarianceTerm() {
}

CSingleTraitTerm::CSingleTraitTerm(const MatrixXd& K):AVarianceTerm() {
	this->setK(K);
}

CSingleTraitTerm::~CSingleTraitTerm() {
}

void CSingleTraitTerm::setScales(const VectorXd& scales) throw(CGPMixException)
{
	if (K_mode==-1)
		throw CGPMixException("CSingleTraitTerm: K needs to be set!");
	this->Kcf->setParams(scales);
}

void CSingleTraitTerm::agetScales(VectorXd* out) const throw(CGPMixException)
{
	if (K_mode==-1)
		throw CGPMixException("CSingleTraitTerm: K needs to be set!");
	static_pointer_cast<CTrait>(this->Kcf)->agetScales(out);
}

muint_t CSingleTraitTerm::getNumberScales() const throw(CGPMixException)
{
	if (K_mode==-1)
		throw CGPMixException("CSingleTraitTerm: K needs to be set!");
	return this->Kcf->getNumberParams();
}

void CSingleTraitTerm::agetTraitCovariance(MatrixXd *out) const throw(CGPMixException)
{
	throw CGPMixException("CSingleTraitTerm: TraitCovariance is not defined");
}

void CSingleTraitTerm::agetVariances(VectorXd* out) const throw(CGPMixException)
{
	this->agetScales(out);
	(*out)=(*out).unaryExpr(std::bind2nd( std::ptr_fun<double,double,double>(pow), 2) );
}


void CSingleTraitTerm::initTerm() throw(CGPMixException)
{
	if (K_mode==-1)
		throw CGPMixException("CSingleTraitTerm: K needs to be set!");
	if (this->Kcf->getParams().rows()!=this->Kcf->getNumberParams())
		this->setScales(randn(Kcf->getNumberParams(),(muint_t)1));
		//TODO: set the seed
	this->is_init=(bool)1;
}

PCovarianceFunction CSingleTraitTerm::getCovariance() const throw(CGPMixException)
{
	if (!is_init)	throw CGPMixException("CSingleTraitTerm: the term is not initialised!");
	return this->Kcf;
}

/* CMultiTraitTerm */

CMultiTraitTerm::CMultiTraitTerm(muint_t P):AVarianceTerm()
{
	this->P=P;
	this->TCtype="Null";
}

CMultiTraitTerm::CMultiTraitTerm(muint_t P, std::string TCtype, const MatrixXd& K):AVarianceTerm()
{
	this->P=P;
	this->setTraitCovarianceType(TCtype);
	this->setK(K);
}

CMultiTraitTerm::CMultiTraitTerm(muint_t P, std::string TCtype, muint_t p, const MatrixXd& K):AVarianceTerm()
{
	this->P=P;
	this->setTraitCovarianceType(TCtype,p);
	this->setK(K);
}

CMultiTraitTerm::~CMultiTraitTerm()
{
}

void CMultiTraitTerm::setTraitCovarianceType(std::string TCtype, muint_t p) throw(CGPMixException)
{
	if(TCtype=="FreeForm")			traitCovariance = PTFreeForm(new CTFreeForm(this->P));
	else if (TCtype=="Block")		traitCovariance = PTFixed(new CTFixed(this->P,MatrixXd::Ones(P,P)));
	else if (TCtype=="Identity")	traitCovariance = PTFixed(new CTFixed(this->P,MatrixXd::Identity(P,P)));
	else if (TCtype=="Dense")		traitCovariance = PTDense(new CTDense(this->P));
	else if (TCtype=="Diagonal")	traitCovariance = PTDiagonal(new CTDiagonal(this->P));
	else if (TCtype=="Full")		traitCovariance = PTLowRank(new CTLowRank(this->P));
	else if (TCtype=="Specific") {
		if (p>=this->P)		throw CGPMixException("CMultiTraitTerm: specify a valid phenotype number");
		MatrixXd K0 = MatrixXd::Zero(P,P);
		K0(p,p) = 1;
		traitCovariance = PTFixed(new CTFixed(this->P,K0));
		std::ostringstream temp;
		temp<<"Specific:pheno"<<p;
		TCtype=temp.str();
	}
	else throw CGPMixException("CMultiTraitTerm: Non valid type!");
	this->TCtype=TCtype;
}

std::string CMultiTraitTerm::getTraitCovarianceType() const throw(CGPMixException)
{
	if (TCtype=="Null")
		throw CGPMixException("CMultiTraitTerm: traitCovariance needs to be set!");
	return this->TCtype;
}

void CMultiTraitTerm::setScales(const VectorXd& scales) throw(CGPMixException)
{
	if (TCtype=="Null")
		throw CGPMixException("CMultiTraitTerm: traitCovariance needs to be set!");
	this->traitCovariance->setParams(scales);
}

void CMultiTraitTerm::agetScales(VectorXd* out) const throw(CGPMixException)
{
	if (TCtype=="Null")
		throw CGPMixException("CMultiTraitTerm: traitCovariance needs to be set!");
	this->traitCovariance->agetScales(out);
}

muint_t CMultiTraitTerm::getNumberScales() const throw(CGPMixException)
{
	if (TCtype=="Null")
		throw CGPMixException("CMultiTraitTerm: traitCovariance needs to be set!");
	return this->traitCovariance->getNumberParams();
}

void CMultiTraitTerm::agetTraitCovariance(MatrixXd *out) const throw(CGPMixException)
{
	if (TCtype=="Null")
		throw CGPMixException("CMultiTraitTerm: traitCovariance needs to be set!");
	if (this->traitCovariance->getParams().rows()!=this->traitCovariance->getNumberParams())
		throw CGPMixException("CMultiTraitTerm: parameters need to be set!");
	this->traitCovariance->aK(out);
}

void CMultiTraitTerm::agetVariances(VectorXd* out) const throw(CGPMixException)
{
	MatrixXd traitCov;
	this->agetTraitCovariance(&traitCov);
	(*out) = traitCov.diagonal();
}

void CMultiTraitTerm::initTerm() throw(CGPMixException)
{
	if (K_mode==-1)
		throw CGPMixException("CSingleTraitTerm: K needs to be set!");
	if (TCtype=="Null")
		throw CGPMixException("CMultiTraitTerm: traitCovariance needs to be set!");
	// IntraTrait Covariance Matrix
	if (K_mode==-1)
		throw CGPMixException("CMultiTraitTerm: K needs to be set!");
	Kcf->setParams(VectorXd::Ones(1));
	Kcf->setParamMask(VectorXd::Zero(1));
	// InterTrait Covariance Matrix
	if (this->traitCovariance->getParams().rows()!=this->traitCovariance->getNumberParams())
		this->setScales(randn(traitCovariance->getNumberParams(),(muint_t)1));
	covariance = PKroneckerCF(new CKroneckerCF(traitCovariance,Kcf));
	this->is_init=(bool)1;
}

PCovarianceFunction CMultiTraitTerm::getCovariance() const throw(CGPMixException)
{
	if (!is_init)	throw CGPMixException("CMultiTraitTerm: the term is not initialised!");
	return this->covariance;
}


/* CMultiTraitTerm */

CVarianceDecomposition::CVarianceDecomposition(const MatrixXd& pheno){
	this->setPheno(pheno);
}

CVarianceDecomposition::~CVarianceDecomposition(){
}

void CVarianceDecomposition::clear()
{
	this->fixedEffs.clear();
	this->terms.clear();
	this->is_init=(bool)0;
}

void CVarianceDecomposition::addFixedEffTerm(const std::string fixedType) throw(CGPMixException)
{
	MatrixXd singleTraitFixed = MatrixXd::Ones(this->N,1);
	this->addFixedEffTerm(fixedType,singleTraitFixed);
}

void CVarianceDecomposition::addFixedEffTerm(const std::string fixedType, const MatrixXd& singleTraitFixed) throw(CGPMixException)
{
	if ((muint_t)singleTraitFixed.cols()!=(muint_t)1 || (muint_t)singleTraitFixed.rows()!=this->N)
		throw CGPMixException("CVarianceDecomposition: the single trait fixed effect has incompatible shape");
	if (fixedType=="common") {
		MatrixXd fixed=MatrixXd::Zero(this->N,this->P);
		for (muint_t p=0; p<this->P; p++)
			fixed.block(0,p,N,1)=singleTraitFixed;
		fixedEffs.push_back(fixed);
	}
	else if (fixedType=="specific") {
		for (muint_t p=0; p<this->P; p++) {
			MatrixXd fixed=MatrixXd::Zero(this->N,this->P);
			fixed.block(0,p,N,1)=singleTraitFixed;
			fixedEffs.push_back(fixed);
		}
	}
	else
		throw CGPMixException("CVarianceDecomposition: non valid fixed effect type");
}

void CVarianceDecomposition::addFixedEffTerm(const MatrixXd& fixed) throw(CGPMixException)
{
	if (fixed.cols()!=this->P || fixed.rows()!=this->N)
		throw CGPMixException("CVarianceDecomposition: the fixed effect has incompatible shape");
	fixedEffs.push_back(fixed);
}

void CVarianceDecomposition::getFixedEffTerm(MatrixXd *out, const muint_t i) const throw(CGPMixException)
{
	if (i>=this->getNumberFixedEffs())
		throw CGPMixException("CVarianceDecomposition: value out of range");
	(*out)=this->fixedEffs[i];
}

void CVarianceDecomposition::clearFixedEffs()
{
	this->fixedEffs.clear();
}

muint_t CVarianceDecomposition::getNumberFixedEffs() const
{
	return (muint_t)(this->fixedEffs.size());
}

void CVarianceDecomposition::setPheno(const MatrixXd& pheno) throw(CGPMixException)
{
	// Set Phenoa and dimensions
	this->pheno = pheno;
	this->N = (muint_t)pheno.rows();
	this->P = (muint_t)pheno.cols();
	// Set all other values to default
	this->clear();
}

void CVarianceDecomposition::getPheno(MatrixXd *out) const throw(CGPMixException)
{
	(*out)=this->pheno;
}

void CVarianceDecomposition::addTerm(PVarianceTerm term) throw(CGPMixException)
{

	if (term->getName()=="CMultiTraitTerm")
		if (term->getNumberTraits()!=this->P)
			throw CGPMixException("CVarianceDecomposition: the term has incompatible number of traits");
		if (term->getNumberIndividuals()!=this->N)
			throw CGPMixException("CVarianceDecomposition: the term has incompatible number of individual");
	else if (term->getName()=="CSingleTraitTerm")
		if (term->getNumberIndividuals()!=this->N*this->P)
			throw CGPMixException("CVarianceDecomposition: the single trait term must have dimensions NP");
	terms.push_back(term);
}

void CVarianceDecomposition::addTerm(const MatrixXd& K) throw(CGPMixException)
{
//TODO
}

void CVarianceDecomposition::addTerm(std::string TCtype, const MatrixXd& K) throw(CGPMixException)
{
	this->addTerm(TCtype,(muint_t)0,K);
}

void CVarianceDecomposition::addTerm(std::string TCtype, muint_t p, const MatrixXd& K) throw(CGPMixException)
{
	if (TCtype=="SingleTrait")
		this->addTerm(K);
	else {
		this->addTerm(PMultiTraitTerm(new CMultiTraitTerm(this->P,TCtype,p,K)));
	}
}

PVarianceTerm CVarianceDecomposition::getTerm(muint_t i) const throw(CGPMixException)
{
	if (i>=this->getNumberTerms())
		throw CGPMixException("CVarianceDecomposition: value out of range");
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

void CVarianceDecomposition::setScales(const VectorXd& scales) const throw(CGPMixException)
{
	if (this->is_init==0)
		throw CGPMixException("CVarianceDecomposition: CVarianceDecomposition needs to be initialised");
	this->covar->setParams(scales);
}

void CVarianceDecomposition::setScales(muint_t i,const VectorXd& scales) const throw(CGPMixException)
{
	if (i>=this->getNumberTerms())
		throw CGPMixException("CVarianceDecomposition: value out of range");
	this->terms[i]->setScales(scales);
}

void CVarianceDecomposition::agetScales(muint_t i, VectorXd* out) const throw(CGPMixException)
{
	if (i>=this->getNumberTerms())
		throw CGPMixException("CVarianceDecomposition: value out of range");
	this->terms[i]->agetScales(out);
}

void CVarianceDecomposition::agetScales(VectorXd* out) throw(CGPMixException)
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

muint_t CVarianceDecomposition::getNumberScales() throw(CGPMixException)
{
	muint_t out=0;
	for(PVarianceTermVec::iterator iter = this->terms.begin(); iter!=this->terms.end();iter++)
	{
		PVarianceTerm term = iter[0];
		out+=term->getNumberScales();
	}
	return out;
}

void CVarianceDecomposition::agetTraitCovariance(muint_t i, MatrixXd *out) const throw(CGPMixException)
{
	if (i>=this->getNumberTerms())
		throw CGPMixException("CVarianceDecomposition: value out of range");
	this->terms[i]->agetTraitCovariance(out);
}

void CVarianceDecomposition::agetVariances(muint_t i, VectorXd* out) const throw(CGPMixException)
{
	if (i>=this->getNumberTerms())
		throw CGPMixException("CVarianceDecomposition: value out of range");
	if (this->terms[i]->getName()=="CSingleTraitTerm") {
		this->terms[i]->agetVariances(out);
		(*out)=(*out)(1,1)*(MatrixXd::Ones(this->P,1));
	}
	else this->terms[i]->agetVariances(out);
}

void CVarianceDecomposition::agetVariances(MatrixXd* out) throw(CGPMixException)
{
	(*out).resize(this->getNumberTerms(),this->P);
	muint_t row=0;
	for(PVarianceTermVec::iterator iter = this->terms.begin(); iter!=this->terms.end();iter++)
	{
		PVarianceTerm term = iter[0];
		VectorXd var_i;
		term->agetVariances(&var_i);
		(*out).block(row,0,1,this->P)=var_i.transpose();
		row++;
	}
}

void CVarianceDecomposition::agetVarComponents(MatrixXd* out) throw(CGPMixException)
{
	agetVariances(out);
	for (muint_t p=0; p<this->P; p++)	(*out).block(0,p,this->getNumberTerms(),1)=(*out).block(0,p,this->getNumberTerms(),1)/(*out).block(0,p,this->getNumberTerms(),1).sum();
}


void CVarianceDecomposition::initGP() throw(CGPMixException)
{
	covar = PSumCF(new CSumCF());
	// Init Covariances and sum them
	for(PVarianceTermVec::iterator iter = this->terms.begin(); iter!=this->terms.end();iter++)
	{
		PVarianceTerm term = iter[0];
		term->initTerm();
		covar->addCovariance(iter[0]->getCovariance());
	}
	//Transform pheno and fixedEffs
	MatrixXd y = MatrixXd::Zero(this->N*this->P,1);
	for (muint_t p=0; p<this->P; p++)	y.block(p*N,0,N,1) = this->pheno.block(0,p,N,1);
	MatrixXd fixed = MatrixXd::Zero(this->N*this->P,this->getNumberFixedEffs());
	muint_t fixedEff_i=0;
	for(MatrixXdVec::const_iterator iter = fixedEffs.begin(); iter!=fixedEffs.end();iter++,fixedEff_i++)
		for (muint_t p=0; p<this->P; p++)	fixed.block(p*N,fixedEff_i,N,1) = iter[0].block(0,p,N,1);
	//Define Likelihood, LinearMean and GP
	PLikNormalNULL lik(new CLikNormalNULL());
	PLinearMean mean(new CLinearMean(y,fixed));
	gp = PGPbase(new CGPbase(covar,lik,mean));
	gp->setY(y);
	//Initialize Params
	CGPHyperParams params;
	params["covar"] = covar->getParams();
	params["dataTerm"] = MatrixXd::Zero(fixed.cols(),1);
	gp->setParams(params);
	opt = PGPopt(new CGPopt(gp));
	this->is_init=1;
}

bool CVarianceDecomposition::trainGP(bool bayes) throw(CGPMixException)
{

	bool conv = false;
	// initGP if is not init
	if (this->is_init==0)	this->initGP();

	// store LML0, scales0
	mfloat_t LML0 = this->getLML();
	VectorXd scales0; this->agetScales(&scales0);

	//train GP
	double time_elapsed=clock();
	conv = this->opt->opt();
	time_elapsed=(clock()-time_elapsed)/ CLOCKS_PER_SEC;

	//check convergence
	conv *= (this->getLMLgrad()<(mfloat_t)1e-6);
	MatrixXd variances;
	this->agetVariances(&variances);
	conv *= (variances.maxCoeff()<(mfloat_t)10.0);

	MatrixXd SigmaTi;
	MatrixXd Sigmai = MatrixXd::Zero((muint_t)this->getNumberScales(),(muint_t)this->getNumberScales());
	MatrixXd Sigma;
	VectorXd filter = this->gp->getParamMask()["covar"];
	if (bayes==true) {
		this->gp->aLMLhess_covar(&SigmaTi);
		muint_t ir=0;
		muint_t ic;
		for (muint_t i=0; i<filter.rows(); i++) {
			ic=0;
			if (filter(i)==1) {
				for (muint_t j=0; j<filter.rows(); j++) {
					if (filter(j)==1) {
						Sigmai(ir,ic)=SigmaTi(i,j);
						ic++;
					}
				}
				ir++;
			}
		}
		Sigma = Sigmai.inverse();
	}

	//Store optimum
	MatrixXd convM = MatrixXd::Zero(1,1); if (conv) convM(0,0)=1;
	MatrixXd LML0M(1,1); LML0M(0,0)=LML0;
	MatrixXd LMLM(1,1); LMLM(0,0)=this->getLML();
	MatrixXd LMLgradM(1,1); LMLgradM(0,0)=this->getLMLgrad();
	MatrixXd time_elapsedM(1,1); time_elapsedM(0,0)=time_elapsed;
	MatrixXd posteriorM(1,1);
	VectorXd scales; this->agetScales(&scales);
	MatrixXd varComponents; this->agetVarComponents(&varComponents);
	optimum["conv"]=convM;
	optimum["LML0"]=LML0M;
	optimum["scales0"]=scales0;
	optimum["LML"]=LMLM;
	optimum["LMLgrad"]=LMLgradM;
	optimum["scales"]=scales;
	optimum["variances"]=variances;
	optimum["varComponents"]=varComponents;
	optimum["time_elapsed"]=time_elapsedM;
	if (bayes==true) {
		optimum["covar_scales"]=Sigma;
		optimum["std_scales"]=Sigma.diagonal().unaryExpr(std::ptr_fun(sqrt));
		posteriorM(1,1) = this->getLML()+0.5*this->getNumberScales()*std::log(2*PI)+0.5*std::log(Sigma.determinant());
		optimum["posterior"]=posteriorM;
	}
	return conv;
}

void CVarianceDecomposition::getFixedEffects(VectorXd* out) throw(CGPMixException)
{
	(*out)=this->gp->getParams()["dataTerm"];
}

mfloat_t CVarianceDecomposition::getLML() throw(CGPMixException)
{
	if (!this->is_init)
		throw CGPMixException("CVarianceDecomposition: the term is not initialised!");
	return -1.*this->gp->LML();
}

mfloat_t CVarianceDecomposition::getLMLgrad() throw(CGPMixException)
{
	if (!this->is_init)
		throw CGPMixException("CVarianceDecomposition: the term is not initialised!");
	mfloat_t out = 0;
	// Squared Norm of LMLgrad["covar"]
	VectorXd grad = this->gp->LMLgrad()["covar"];
	VectorXd filter = this->gp->getParamMask()["covar"];
	for (muint_t i=0; i<grad.rows(); i++)
		if (filter(i)==1)	out +=std::pow(grad(i),2);
	// Squared Norm of LMLgrad["dataTerm"]
	grad = this->gp->LMLgrad()["dataTerm"];
	for (muint_t i=0; i<grad.rows(); i++)	out +=std::pow(grad(i),2);
	// Square Root
	out = std::sqrt(out);
	return out;
}



} //end:: namespace
