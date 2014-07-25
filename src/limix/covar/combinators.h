// Copyright(c) 2014, The LIMIX developers(Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef COMBINATORS_H_
#define COMBINATORS_H_

#include <vector>
#include "limix/covar/covariance.h"


namespace limix {

//!> Define a vector of covariances which is needed to represent the sum and product CF
typedef std::vector<PCovarianceFunction> ACovarVec;

class AMultiCF : public ACovarianceFunction
{
protected:
	ACovarVec vecCovariances;
	muint_t numMaxCovariances; //!< internal variance, controlling the maximum number of cavariances in a combinators, this may depend on the CF (default -1) - no limi
public:
	/*!
	 * Constructor for abstract type of MultiPleCovariances
	 * \param covariances: pre-initialized vector with sub covarinaces
	 * \param numMaxCovariances: maximum number of covariances to permit
	 */
	AMultiCF(const ACovarVec& covariances,muint_t numMaxCovariances=999);

	/*!
	 * Constructor for abstract type of MultiPleCovariances
	 * \param numCovariancesInit: number of covarinaces to expect during init
	 * \param numMaxCovariances: maximum number of covariances to permit
	 */
	AMultiCF(muint_t numCovariancesInit=0,muint_t numMaxCovariances=999);

	//destructors
	virtual ~AMultiCF();
	virtual muint_t Kdim() const ;

	//sync stuff
	void addSyncChild(Pbool l);
	void delSyncChild(Pbool l);

	//access to covariance arrays
	void addCovariance(PCovarianceFunction covar) ;
	void setCovariance(muint_t i,PCovarianceFunction covar) ;
	PCovarianceFunction getCovariance(muint_t i) ;

	virtual muint_t getNumberDimensions() const ;
	virtual void setNumberDimensions(muint_t numberDimensions) ;
	virtual muint_t getNumberParams() const;

	//setX and getX
	virtual void setX(const CovarInput& X) ;				//!< setter for covariance input matrix X
	virtual void agetX(CovarInput* Xout) const ;			//!< getter for covariance input matrix X
	/*!
	setter for a single column of CovarInput X
	@param X	single column of a CovarInput
	@param col	index of the column
	*/
	virtual void setXcol(const CovarInput& X,muint_t col) ;

	//set and get Params
	virtual void setParams(const CovarParams& params);
	virtual void agetParams(CovarParams* out) const;
	//set and get Param masks
	virtual void agetParamMask(CovarParams* out) const;
	virtual void setParamMask(const CovarParams& params);
	//get parameter bounds
	virtual void agetParamBounds0(CovarParams* lower,CovarParams* upper) const;
    virtual void agetParamBounds(CovarParams* lower,CovarParams* upper) const;
	virtual void setParamBounds(const CovarParams& lower, const CovarParams& upper)  ;
};

class CSumCF : public AMultiCF {
public:
	CSumCF(const ACovarVec& covariances);
	CSumCF(const muint_t numCovariances=0);
	//destructors
	virtual ~CSumCF();

	//overloaded pure virtual members
	virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const ;
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const ;
	virtual void aKgrad_param(MatrixXd* out,const muint_t i) const ;
    virtual void aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const ;
	virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const ;
	virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const ;
	//optional overloadings:
	virtual void aK(MatrixXd* out) const ;
	virtual void aKdiag(VectorXd* out) const ;
	virtual void aKgrad_X(MatrixXd* out,const muint_t d) const ;
	virtual std::string getName() const;
};
typedef sptr<CSumCF> PSumCF;

/*!
* Kronecker function for pairs of covariances
*/
class CLinCombCF : public AMultiCF {
protected:
	VectorXd coeff;
public:
	CLinCombCF(const ACovarVec& covariances);
	CLinCombCF(const muint_t numCovariances=0);
	//destructors
	virtual ~CLinCombCF();
	//linear coefficients
	virtual void setCoeff(const VectorXd& coeff);
	virtual void agetCoeff(VectorXd* out) const;
	//overloaded pure virtual members
	virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const ;
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const ;
	virtual void aKgrad_param(MatrixXd* out,const muint_t i) const ;
    virtual void aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const ;
	virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const ;
	virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const ;
	//optional overloadings:
	virtual void aK(MatrixXd* out) const ;
	virtual void aKdiag(VectorXd* out) const ;
	virtual void aKgrad_X(MatrixXd* out,const muint_t d) const ;
	virtual std::string getName() const;
};
typedef sptr<CLinCombCF> PLinCombCF;



/*!
Product combinator of a pair of covariances
*/
class CProductCF : public AMultiCF {
public:
	CProductCF(const ACovarVec& covariances);
	CProductCF(const muint_t numCovariances=0);
	//destructors
	virtual ~CProductCF();

	//overloaded pure virtual members
	virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const ;
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const ;
	virtual void aKgrad_param(MatrixXd* out,const muint_t i) const ;
    virtual void aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const ;
	virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const ;
	virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const ;
	//optional overloadings:
	virtual void aK(MatrixXd* out) const ;
	virtual void aKdiag(VectorXd* out) const ;
	virtual void aKgrad_X(MatrixXd* out,const muint_t d) const ;
	virtual std::string getName() const;
};
typedef sptr<CProductCF> PProductCF;


/*
 * Kronecker function for pairs of covariances
*/
    
/*!
 * \brief Kronecker structure combinator for two covariances
 *
 * Combines two covarianes Kc \kron Kr. The class supports both propper Kronecker structures
 * as well as "soft Kronecker, where
 */
class CKroneckerCF: public AMultiCF
{
protected:
	//optional indicator vector to pull together the kronecker structure
	MatrixXi kroneckerIndicator; //!< indicator with kronecker structure N row (total samples), with indexes for [row,col] to create sof Kronecker structures
public:
	CKroneckerCF();
	CKroneckerCF(PCovarianceFunction col,PCovarianceFunction row);
	virtual ~CKroneckerCF();
	virtual muint_t Kdim() const ;

	//Access to vecCovariances
	virtual void setRowCovariance(PCovarianceFunction cov);
	virtual void setColCovariance(PCovarianceFunction cov);
	PCovarianceFunction getRowCovariance() ;
	PCovarianceFunction getColCovariance() ;

	/*!
	 * set KronecekerIndicator, which needs to be N x 2 with indicse for row & column of individual elements
	 */
	void setKroneckerIndicator(const MatrixXi& kroneckerIndicator);

	/*!
	 * getKroneckerIndicator.
	 */
	void getKroneckerIndicator(MatrixXi* out) const;


	/*!
	 * is a kronecker covariance? if a KroneckerIndicator is set this object is not a propper Kronecker.
	 */
	bool isKronecker() const;

	//X handling
	virtual void setX(const CovarInput& X)  {};					//!< No function as Combinator has no X of its own. See .setXr() and .setXc() instead for setters of the row and column covariance inputs.
	virtual void agetX(CovarInput* Xout) const  {};				//!< No function as Combinator has no X of its own. See .getXr() and .getXc() instead for getters of the row and column covariance inputs.
	virtual void setXcol(const CovarInput& X,muint_t col)  {};	//!< No function as Combinator has no X of its own. See .setXr() and .setXc() instead for setters of the row and column covariance inputs.
	virtual void setXr(const CovarInput& Xr) ;	//!< setter of the covariance input of the row covariance
	virtual void setXc(const CovarInput& Xc) ;	//!< setter of the covariance input of the column covariance

	//overloaded pure virtual members
	virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const ;
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const ;
	virtual void aKgrad_param(MatrixXd* out,const muint_t i) const ;
    virtual void aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const ;
	virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const ;
	virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const ;
	//optional overloaded functions for efficiency:
	virtual void aK(MatrixXd* out) const ;
	virtual void aKdiag(VectorXd* out) const ;
	virtual void aKgrad_X(MatrixXd* out,const muint_t d) const ;

	virtual std::string getName() const;

	/*!
	 * Convenience functin to generator a a Kronecker Indicator matrix ([N*P,2])
	 * which is fully kroneckerized. This function is used to perform missing value imputations, etc.
	 * \see CVarianceDecomposition
	 * \param Ncols: number of samples (row covariance)
	 * \param Nrow: number of individuals (col covariance)
	 */
	static void createKroneckerIndex (MatrixXi* out,muint_t Ncols, muint_t Nrows);
};
typedef sptr<CKroneckerCF> PKroneckerCF;

} //end namespace limix




#endif /* COMBINATORS_H_ */
