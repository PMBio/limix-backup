/*
 * CVarianceDecomposition.h
 *
 *  Created on: Nov 27, 2012
 *      Author: stegle
 */

#ifndef CVARIANCEDECOMPOSITION_H_
#define CVARIANCEDECOMPOSITION_H_

#include "limix/types.h"
#include "limix/covar/freeform.h"
#include "limix/covar/combinators.h"
#include "limix/mean/CLinearMean.h"
#include "limix/gp/gp_base.h"
#include "limix/gp/gp_opt.h"

namespace limix {

//typedef std::set<muint_t> muint_set;
//typedef std::set<mfloat_t> mfloat_set;
//typedef std::vector<PVarianceTerm> PVarianceTermVec;

/*
 * CVarainceTerm captures a specific term in the varaince model
 */
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//ignore C++ versions
//%ignore AVarianceTerm::getHpInit;
%rename(getK) AVarianceTerm::agetK;
%rename(getX) AVarianceTerm::agetX;
%rename(getScales) AVarianceTerm::agetScales;
%rename(getTraitCovariance) AVarianceTerm::agetTraitCovariance;
%rename(getVariances) AVarianceTerm::agetVariances;
#endif

class AVarianceTerm {
protected:
	//Covariance Functions
	PCovarianceFunction Kcf;
	//Kinship matrix and SNP data
	muint_t K_mode;
	MatrixXd K;
	MatrixXd X;
	//boold
	bool fitted, is_init;
public:
	AVarianceTerm();
	virtual ~AVarianceTerm();

	virtual std::string getName() = 0;
	virtual std::string getInfo() = 0;

	virtual muint_t getNumberTraits()=0;
	virtual muint_t getNumberIndividuals() const throw(CGPMixException);

	//Kinship and SNP data
	virtual void setK(const MatrixXd& K) throw(CGPMixException);
	virtual void setX(const MatrixXd& X) throw(CGPMixException);
	virtual void agetK(MatrixXd *out) const throw(CGPMixException);
	virtual void agetX(MatrixXd *out) const throw(CGPMixException);

	//Params Handling
	virtual void setScales(const VectorXd& scales) throw(CGPMixException) =0;
	virtual void agetScales(VectorXd* scales) const throw(CGPMixException) =0;
	virtual muint_t getNumberScales() const throw(CGPMixException) = 0;
	//Covariance, Variance, Correlations? and Variance Components
	virtual void agetTraitCovariance(MatrixXd *out) const throw(CGPMixException) =0;
	virtual void agetVariances(VectorXd* out) const throw(CGPMixException) =0;

	//initTerm and getCovariance
	virtual void initTerm() throw(CGPMixException) = 0;
	virtual PCovarianceFunction getCovariance() const throw(CGPMixException) =0;

};
typedef sptr<AVarianceTerm> PVarianceTerm;


/*
 * 	Single trait variance Term
 */

#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
#endif

class CSingleTraitTerm : public AVarianceTerm
{
protected:
public:
	CSingleTraitTerm();
	CSingleTraitTerm(const MatrixXd& K);
	virtual ~CSingleTraitTerm();

	virtual std::string getName() {return "CSingleTraitTerm";};
	virtual std::string getInfo() {return "POPPY";};

	virtual muint_t getNumberTraits() {return (muint_t)1;};

	// Params Handling
	virtual void setScales(const VectorXd& scales) throw(CGPMixException);
	virtual void agetScales(VectorXd* out) const throw(CGPMixException);
	virtual muint_t getNumberScales() const throw(CGPMixException);
	//Covariance, Variance, Correlations? and Variance Components
	virtual void agetTraitCovariance(MatrixXd *out) const throw(CGPMixException);
	virtual void agetVariances(VectorXd* out) const throw(CGPMixException);

	//initTerm and get covariance
	virtual void initTerm() throw(CGPMixException);
	virtual PCovarianceFunction getCovariance() const throw(CGPMixException);
};
typedef sptr<CSingleTraitTerm> PSingleTraitTerm;

/*
 * 	Single trait variance Term
 */

#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
#endif

class CMultiTraitTerm : public AVarianceTerm
{
protected:
	muint_t P;
	PTraitCF traitCovariance;
	std::string TCtype;
	// TOTAL COVARIANCE
	PKroneckerCF covariance;
public:
	CMultiTraitTerm(muint_t P);
	CMultiTraitTerm(muint_t P, std::string TCtype, const MatrixXd& K);
	CMultiTraitTerm(muint_t P, std::string TCtype, muint_t p, const MatrixXd& K);
	virtual ~CMultiTraitTerm();

	virtual std::string getName() {return "CMultiTraitTerm";};
	virtual std::string getInfo() {return "POPPY";};

	virtual muint_t getNumberTraits() {return (muint_t)this->P;};

	//MultiPhenoFramework
	void setTraitCovarianceType(std::string TCtype, muint_t p=0) throw(CGPMixException);
	std::string getTraitCovarianceType() const throw(CGPMixException);

	// Params Handling
	virtual void setScales(const VectorXd& scales) throw(CGPMixException);
	virtual void agetScales(VectorXd* out) const throw(CGPMixException);
	virtual muint_t getNumberScales() const throw(CGPMixException);
	//Covariance, Variance, Correlations? and Variance Components
	virtual void agetTraitCovariance(MatrixXd *out) const throw(CGPMixException);
	virtual void agetVariances(VectorXd* out) const throw(CGPMixException);

	//initTerm and get covariance
	virtual void initTerm() throw(CGPMixException);
	virtual PCovarianceFunction getCovariance() const throw(CGPMixException);
};
typedef sptr<CMultiTraitTerm> PMultiTraitTerm;


/*
 * 	Variance Decomposition
 */

typedef std::vector<PVarianceTerm> PVarianceTermVec;
typedef std::vector<MatrixXd> MatrixXdVec;
typedef CGPHyperParams CVDOptimum;

#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
%rename(getScales) CVarianceDecomposition::agetScales;
%rename(getVariances) CVarianceDecomposition::agetVariances;
%rename(getTraitCovariance) CVarianceDecomposition::agetTraitCovariance;
%rename(getVarComponents) CVarianceDecomposition::agetVarComponents;
#endif

class CVarianceDecomposition
{
protected:
	//Dimensions
	muint_t N,P;
	//Fixed effects
	MatrixXdVec fixedEffs;
	//terms
	PVarianceTermVec terms;
	//Gaussian Process
	PGPbase gp;
	PGPopt opt;
	//Total Covariance
	PSumCF covar;
	//pheno
	MatrixXd pheno;
	//Optimum
	CVDOptimum optimum;
	//bool
	bool is_init;
public:
	CVarianceDecomposition(const MatrixXd& pheno);
	~CVarianceDecomposition();

	//Delete All Fixed and Variance Terms
	virtual void clear();
	//Phenotype
	virtual void setPheno(const MatrixXd& pheno) throw(CGPMixException);
	virtual void getPheno(MatrixXd *out) const throw(CGPMixException);
	virtual muint_t getNumberTraits() const throw(CGPMixException) {return this->P;};
	virtual muint_t getNumberIndividuals() const throw(CGPMixException) {return this->N;};

	// Fixed Effect Terms
	virtual void addFixedEffTerm(const std::string fixedType) throw(CGPMixException);
	virtual void addFixedEffTerm(const std::string fixedType, const MatrixXd& singleTraitFixed) throw(CGPMixException);
	virtual void addFixedEffTerm(const MatrixXd& fixed) throw(CGPMixException);
	virtual void getFixedEffTerm(MatrixXd *out, const muint_t i) const throw(CGPMixException);
	virtual void clearFixedEffs();
	virtual muint_t getNumberFixedEffs() const;

	// Variance Term
	virtual void addTerm(PVarianceTerm term) throw(CGPMixException);
	virtual void addTerm(const MatrixXd& K) throw(CGPMixException);	//add Single Trait Term
	virtual void addTerm(std::string TCtype, const MatrixXd& K) throw(CGPMixException);	//add Multi Trait Term
	virtual void addTerm(std::string TCtype, muint_t p, const MatrixXd& K) throw(CGPMixException);	//add Multi Trait Term
	virtual PVarianceTerm getTerm(muint_t i) const throw(CGPMixException);
	virtual void clearTerms();
	virtual muint_t getNumberTerms() const;
	// access Variance Term Features
	virtual void setScales(const VectorXd& scales) const throw(CGPMixException);
	virtual void setScales(muint_t i,const VectorXd& scales) const throw(CGPMixException);
	virtual void agetScales(muint_t i, VectorXd* out) const throw(CGPMixException);
	virtual void agetScales(VectorXd* out) throw(CGPMixException);
	virtual muint_t getNumberScales() throw(CGPMixException);
	virtual void agetTraitCovariance(muint_t i, MatrixXd *out) const throw(CGPMixException);
	virtual void agetVariances(muint_t i, VectorXd* out) const throw(CGPMixException);
	virtual void agetVariances(MatrixXd* out) throw(CGPMixException);
	virtual void agetVarComponents(MatrixXd* out) throw(CGPMixException);

	//get gp, Covar and mean
	virtual PGPbase getGP() {return this->gp;};
	virtual PSumCF getCovar() {return this->covar;};
	virtual PLinearMean getMean() {return static_pointer_cast<CLinearMean>(this->gp->getDataTerm());};
	//init and train GP
	virtual void initGP() throw(CGPMixException);
	virtual bool trainGP(bool bayes=false) throw(CGPMixException);
	virtual CVDOptimum getOptimum() {return this->optimum;};
	//access GP
	virtual void getFixedEffects(VectorXd* out) throw(CGPMixException);
	virtual mfloat_t getLML() throw(CGPMixException);
	virtual mfloat_t getLMLgrad() throw(CGPMixException);
};
typedef sptr<CVarianceDecomposition> PVarianceDecomposition;


} //end: namespace limix

#endif /* CVARIANCEDECOMPOSITION_H_ */
