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
class AVarianceTerm {
protected:
	//Covariance Functions
	PFixedCF Kcf;
	//Kinship matrix and SNP data
	bool Knull;
	MatrixXd K;
	MatrixXd X;
	//boold
	bool fitted, is_init;
public:
	AVarianceTerm();
	virtual ~AVarianceTerm();

	virtual std::string getName() = 0;
	virtual std::string getInfo() = 0;

	/*!
	 * set a filter with individuals
	 * This functionality is typically used to filter out non-observed individuals
	 */
	virtual void setSampleFilter(const MatrixXb& filter)  = 0;

	virtual muint_t getNumberTraits()=0;
	virtual muint_t getNumberIndividuals() const ;

	//Kinship and SNP data
	virtual void setK(const MatrixXd& K) ;
	virtual void agetK(MatrixXd *out) const ;
	virtual PFixedCF getKcf() {return Kcf;};

	virtual PCovarianceFunction getTraitCovar() const  =0;

	//Params Handling
	virtual void setScales(const VectorXd& scales)  =0;
	virtual void agetScales(VectorXd* scales) const  =0;
	virtual muint_t getNumberScales() const  = 0;

	//initTerm and getCovariance
	virtual void initTerm()  = 0;
	virtual PCovarianceFunction getCovariance() const  =0;

};
typedef sptr<AVarianceTerm> PVarianceTerm;


/*
 * 	Single trait variance Term
 */


class CSingleTraitTerm : public AVarianceTerm
{
protected:
public:
	CSingleTraitTerm();
	CSingleTraitTerm(const MatrixXd& K);
	virtual ~CSingleTraitTerm();

	virtual std::string getName() {return "CSingleTraitTerm";};
	virtual std::string getInfo() {return "POPPY";};

	virtual void setSampleFilter(const MatrixXb& filter) ;



	virtual muint_t getNumberTraits() {return (muint_t)1;};


	virtual PCovarianceFunction getTraitCovar() const ;

	// Params Handling
	virtual void setScales(const VectorXd& scales) ;
	virtual void agetScales(VectorXd* out) const ;
	virtual muint_t getNumberScales() const ;

	//initTerm and get covariance
	virtual void initTerm() ;
	virtual PCovarianceFunction getCovariance() const ;
};
typedef sptr<CSingleTraitTerm> PSingleTraitTerm;

/*
 * 	Single trait variance Term
 */

class CMultiTraitTerm : public AVarianceTerm
{
protected:
	muint_t P;
	PCovarianceFunction traitCovariance;
	bool isNull;
	// TOTAL COVARIANCE
	PKroneckerCF covariance;
public:
	CMultiTraitTerm(muint_t P);
	CMultiTraitTerm(muint_t P, PCovarianceFunction traitCovar, const MatrixXd& K);
	virtual ~CMultiTraitTerm();

	virtual std::string getName() {return "CMultiTraitTerm";};
	virtual std::string getInfo() {return "POPPY";};

	/*!
	 * sets a filtr of observed individuals.
	 * Note, this filter breaks the Kronecker strucutre and is only applicable in a non-Kronecker settings
	 */
	virtual void setSampleFilter(const MatrixXb& filter) ;


	virtual muint_t getNumberTraits() {return (muint_t)this->P;};

	//MultiPhenoFramework
	void setTraitCovar(PCovarianceFunction traitCovar) ;
	virtual PCovarianceFunction getTraitCovar() const ;

	// Params Handling
	virtual void setScales(const VectorXd& scales) ;
	virtual void agetScales(VectorXd* out) const ;
	virtual muint_t getNumberScales() const ;

	//initTerm and get covariance
	virtual void initTerm() ;
	virtual PCovarianceFunction getCovariance() const ;
};
typedef sptr<CMultiTraitTerm> PMultiTraitTerm;


/*
 * 	Variance Decomposition
 */

typedef std::vector<PVarianceTerm> PVarianceTermVec;
typedef std::vector<MatrixXd> MatrixXdVec;

class CVarianceDecomposition
{
protected:
	//Dimensions
	muint_t N,P;
	//Fixed effects
	MatrixXdVec fixedEffs;
	MatrixXdVec designs;
	//terms
	PVarianceTermVec terms;
	//Gaussian Process
	PGPbase gp;
	PGPopt opt;
	//Total Covariance
	PSumCF covar;
	//pheno
	MatrixXd pheno;
	//missing values in pheno
	MatrixXb phenoNAN; //<<! indicator with possible missing values in pheno
	bool phenoNANany; //<<! any phenotype entry missing?
	//bool
	bool is_init;
	bool fast;
public:
	CVarianceDecomposition(const MatrixXd& pheno);
	~CVarianceDecomposition();

	//Delete All Fixed and Variance Terms
	virtual void clear();
	//Phenotype
	virtual void setPheno(const MatrixXd& pheno) ;
	virtual void getPheno(MatrixXd *out) const ;
	virtual muint_t getNumberTraits() const  {return this->P;};
	virtual muint_t getNumberIndividuals() const  {return this->N;};

	// Fixed Effect Terms
	virtual void addFixedEffTerm(const MatrixXd& design, const MatrixXd& fixed) ;
	virtual void addFixedEffTerm(const MatrixXd& F) ;
	virtual void getFixed(MatrixXd *out, const muint_t i) const ;
	virtual void getDesign(MatrixXd *out, const muint_t i) const ;
	virtual void clearFixedEffs();
	virtual muint_t getNumberFixedEffs() const;
	// access fixed

	// Variance Term
	virtual void addTerm(PVarianceTerm term) ;
	virtual void addTerm(const MatrixXd& K) ;	//add Single Trait Term
	virtual void addTerm(PCovarianceFunction traitCovar, const MatrixXd& K) ;	//add Multi Trait Term
	virtual PVarianceTerm getTerm(muint_t i) const ;
	virtual void clearTerms();
	virtual muint_t getNumberTerms() const;
	// access Variance Term Features
	virtual void setScales(const VectorXd& scales) const ;
	virtual void setScales(muint_t i,const VectorXd& scales) const ;
	virtual void agetScales(muint_t i, VectorXd* out) const ;
	virtual void agetScales(VectorXd* out) ;
	virtual muint_t getNumberScales() ;

	//get gp, Covar and mean
	virtual PGPbase getGP() {return this->gp;};
	virtual PSumCF getCovar() {return this->covar;};
	virtual PLinearMean getMean() {return static_pointer_cast<CLinearMean>(this->gp->getDataTerm());};
	//init and train GP
	virtual void initGPparams() ;
	virtual void initGP(bool fast=false) ;
	virtual void initGPbase() ;
	virtual void initGPkronSum() ;
	virtual bool trainGP() ;
	//access GP
	virtual void getFixedEffects(VectorXd* out) ;
	virtual mfloat_t getLML() ;
	virtual mfloat_t getLMLgrad() ;
	virtual mfloat_t getLMLgradGPbase() ;
	virtual mfloat_t getLMLgradGPkronSum() ;

	// estimate single trait heritability
	static void aestimateHeritability(VectorXd* out, const MatrixXd& Y, const MatrixXd& fixed, const MatrixXd& K);

};
typedef sptr<CVarianceDecomposition> PVarianceDecomposition;


} //end: namespace limix

#endif /* CVARIANCEDECOMPOSITION_H_ */
