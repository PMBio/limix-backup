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

#ifndef ACOVARIANCE_H_
#define ACOVARIANCE_H_

#include "limix/types.h"
namespace limix {

//define covariance function types

typedef MatrixXd CovarInput;
typedef VectorXd CovarParams;




/*! \brief Abstract base class for covariance functions
 *
 * Covariance functions generate kernel matrices from inputs (\see X) and a set of hyperparameters (\see params).
 * This abstract base class provides the interface and standard implementation for convenience function.
 */
class ACovarianceFunction : public CParamObject {
protected:
	CovarInput X; 			//!<Input of the covariance function/kernel
	CovarParams params; 	//!<Local covariance parameters (the hyperparameters of K)
	CovarParams paramsMask; //!<mask of the covariance hyperparameters for optimization
	muint_t numberParams;   //!<number of parameters
	muint_t numberDimensions;//!<dimension of the expected input (\see X)
	CovarParams bound_lower; //!<lower bound of hyperparameters for optimization
	CovarParams bound_upper; //!<<upper bound of hyperparameters for optimization

	//helper functions:
	/*!
	 * check that the the index is within the permitted dimensions
	 */
	inline void checkWithinDimensions(muint_t d) const ;
	/*!
	 * check that the the index is within the permitted dimensions
	 */
	inline void checkWithinParams(muint_t i) const ;
	/*!
	 * check that the covariance input X has the matching dimensions
	 */
	inline void checkXDimensions(const CovarInput& X) const ;
	/*!
	 * check that the covariance parameters has the matching dimensions
	 */
	inline void checkParamDimensions(const CovarParams& params) const ;
	/*!
	 *\brief return the parameter Mask as out object.
	 *
	 * Note ParamMask0 is the intrinsic constraint of parameters that never need opimization
	 */
	virtual void agetParamMask0(CovarParams* out) const;

	/*!
	 * initialize parameter vector
	 */
	virtual void initParams();

public:
	/*! Covariance functions require knowledge on the number of parameters which is constant thorough their lifetie
	 *
	 */
	ACovarianceFunction(const muint_t numberParams=0);
	//destructors
	virtual ~ACovarianceFunction();

	/*!
	 * get a readable name of the covariance function.
	 * Note: usually these are type names, and hence if multiple covarainces are combined these are not unique
	 */
	virtual std::string getName() const = 0;

	/*!
	 * set the hyperparameters
	 */
	virtual void setParams(const CovarParams& params);
	/*!
	 * get current hyperparameters (argout variant)
	 * \param out points to the outargument
	 */
	virtual void agetParams(CovarParams* out) const;
	/*!
	 * get the internal bounds of the covariance that cannot be overwritten.
	 * Typically, these parameters need to be positive to ensure that the covariance is valid, etc.
	 */
	virtual void agetParamBounds0(CovarParams* lower,CovarParams* upper) const;
	/*!
	 * In addition to the intrinsic bounds, there may be userdefined bounds. This function returns the combination
	 * of userdefined and intrinsic bounds.
	 */
	virtual void agetParamBounds(CovarParams* lower,CovarParams* upper) const;
	/*!
	 * set Userdefined bounds. These are combined with the intrinsic bounds (see @agetParamBounds0).
	 */
	virtual void setParamBounds(const CovarParams& lower, const CovarParams& upper)  ;
	/*!
	 * return the parameter mask for optimization(argout, see also @getParamMask)
	 */
	virtual void agetParamMask(CovarParams* out) const;
	/*!
	 * return parameter mask (pointer version, see also @agetParamMask)
	 */
	virtual CovarParams getParamMask() const;

	/*!
	 * set the parameter mask for parameter optimization.
	 * \param params has the same dimension than then hpyerparameters and indicates whether the elements are to be optimized (=0: no optimization, =1: included in optimization)
	 */
	virtual void setParamMask(const CovarParams& params);

	/*!
	 * set new X value
	 * \param is an N x D matrix. The dimensionality needs to match the internal dimensnion (\sa getDimX).
	 */
	virtual void setX(const CovarInput& X) ;
	/*!
	 * set a column of X
	 * \param X column matri N x 1
	 * \param col is the column to be set (0..D)
	 */
	virtual void setXcol(const CovarInput& X, muint_t col) ;

	/*!
	 * get the current X (argout version)
	 * \param out denotes the out argument
	 */
	virtual void agetX(CovarInput* Xout) const ;
	/*!
	 * get the dimensionality of the input
	 */
	virtual muint_t getDimX() const {return (muint_t)(this->X.cols());}
	/*!
	 * get the number of parameters this covaraince function has
	 */
	virtual muint_t getNumberParams() const;
	/*!
	 * get the number of dimensions
	 * TOOD: what is the difference between this and getDimX?
	 */
	virtual muint_t getNumberDimensions() const;
	/*!
	 * set the number of input dimensions
	 * \param numberDiemensions >0 and defines the dimensionality of the input
	 */
	virtual void setNumberDimensions(muint_t numberDimensions);

	/*!
	 * TODO?
	 */
	virtual muint_t Kdim() const ;

	/*!
	 * calculate the covaraince matrix
	 * \param out points to the target matrix
	 */
	virtual void aK(MatrixXd* out) const ;
	/*!
	 * calculate the diagonal of the covariance matrix
	 * \param out points to the target matrix
	 */
	virtual void aKdiag(VectorXd* out) const ;
	/*!
	 * \brief calculate the gradient of the covariance with respect to all entries X[:,d]
	 * \param out points to the target matrix
	 * \param d denotes the column of X
	 *
	 * Note: formally, the gradients of K w.r.t any hyparparameter is again an NxN matrix (see for example @aKgrad_param).
	 * However, if the gradient is w.r.t to a particular input X_{n,d}, only the nth row in that gradient matrix is non-zero.
	 * To safe computation, the function calculates the gradient with respect to all X[:,d] simultaneously where the entries reflect the
	 * non-zero row in each.
	 */
	virtual void aKgrad_X(MatrixXd* out,const muint_t d) const ;

	/*!
	 * calculate the cross covariance K(X,X*)
	 * \param out points to the target matrix
	 * \param Xstar is the test covriance matrix X*
	 */
	virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const  = 0;
	/*!
	 * calculate the diagonal of the self covariance K(X*,X*)
	 * \param out points to the target matrix
	 * \param Xstar is the test covariance matrix X*
	 */
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const  = 0;
	/*!
	 * calculate the gradient of the covariance w.r.t. parameter i
	 * \prarm out points to the target matrix (N x N)
	 * \param i denotes the index of the hyperparameter (see @setParams and @getParams)
	 */
	virtual void aKgrad_param(MatrixXd* out,const muint_t i) const  =0;
	/*!
	 * calculate the second derivative of the covariance matrix w.r.t. parameter i and j
	 * \param out points to the target matrix (N x N)
	 * \param i is the first derivative index
	 * \param j is the second derivative index
	 */
    virtual void aKhess_param(MatrixXd* out,const muint_t i,const muint_t j) const  =0;
    /*!
     *calculate the gradient of the cross covariance K(X,X*) w.r.t. to a column of X*
     * \param out points to the target matrix (N x N')
     * \param Xstar is the test covariance
     * \param d is the dimension
     */
	virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const  = 0;
	/*!
	 * calculate the diagonal of the graident matrix w.r.t. X
	 *\param out poinst to the target marix (N x N)
	 *\param d denotes the dimension of X
	 */
	virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const  = 0;

	//Inline convenience functions:
	/*!
	 * \see aK
	 */
	inline MatrixXd K() const;
	/*!
	 * \see agetParams
	 */
	virtual CovarParams getParams() const;
	/*!
	 * \see agetX
	 */
	inline CovarInput getX() const;
	/*!
	 * \see aKdiag()
	 */
	inline VectorXd Kdiag() const;
	/*!
	 * \see aKcross
	 */
	inline MatrixXd Kcross( const CovarInput& Xstar ) const ;
	/*!
	 * \see aKcross_diag
	 */
	inline VectorXd Kcross_diag(const CovarInput& Xstar) const ;
	/*!
	 * \see aKgrad_param
	 */
	inline MatrixXd Kgrad_param(const muint_t i) const ;
	/*!
	 * \see aKhess_param
	 */
    inline MatrixXd Khess_param(const muint_t i, const muint_t j) const ;
	/*!
	 * \see aKcross_grad_X
	 */
	inline MatrixXd Kcross_grad_X(const CovarInput& Xstar, const muint_t d) const ;
	/*!
	 * \see aKgrad_X
	 */
	inline MatrixXd Kgrad_X(const muint_t d) const ;
	/*!
	 * \see aKdiag_grad_X
	 */
	inline VectorXd Kdiag_grad_X(const muint_t d) const ;

	/* Static methods*/
	/*!
	 * check gradient of given covariance function with respect to hyperparameters, comparing with a numerical equivalent
	 * \param covar is the covariance function to be checked
	 * \param relchange is the step size in the numerical calculation
	 * \param threshold is the threshold of concordance.
	 */
	static bool check_covariance_Kgrad_theta(ACovarianceFunction& covar,mfloat_t relchange=1E-5,mfloat_t threshold=1E-2);
	/*!
	 * check gradient of given covariance function with respect to input x, comparing with a numerical equivalent
	 * \param covar is the covariance function to be checked
	 * \param relchange is the step size in the numerical calculation
	 * \param threshold is the threshold of concordance.
	 * \param check_diag denotes whether the diagonal gradients are to be tested separately
	 */
	static bool check_covariance_Kgrad_x(ACovarianceFunction& covar,mfloat_t relchange=1E-5,mfloat_t threshold=1E-2,bool check_diag=true);
    //numerical hessian
	/*!
	 * calculates a numerical hessian for covar
	 * \param covar is the covariance function to be checked
	 * \param out denotes the target matrix
	 * \param i is the first parameter index
	 * \param j is the second parameter index
	 */
    static void aKhess_param_num(ACovarianceFunction& covar, MatrixXd* out, const muint_t i, const muint_t j) ;
	//covariance normalization
	//template <typename Derived1, typename Derived2,typename Derived3,typename Derived4,typename Derived5>
	//inline static void scale_K(const Eigen::MatrixBase<Derived1> & K_);

	friend std::ostream& operator<< (std::ostream &out, ACovarianceFunction &ACovarianceFunction);
};
typedef sptr<ACovarianceFunction> PCovarianceFunction;


/*!
 * \brief Depricated: Cache object which inherits all the interface functions from Covar
 *
 * DEPRICATED. OLD CACHE FUNCTIONS FOR COVARIANCES CACHES
 * The main purpose of this function is to precompute K and suitable decomposition.
 * Support is provided for both, cholesky-based decompositions (\sa rgetCholK) and symmetric eigen decompositions (\sa rgetUK, \sa rgetUS)
 */
class CCovarianceFunctionCacheOld : public CParamObject
{
protected:
	//wrapper covariance Function
	PCovarianceFunction covar;
	MatrixXd KCache; //!< cached covaraince matrix
	//SVD
	MatrixXd UCache; //!< cached eigen vectors (K=USU^T)
	VectorXd SCache; //!< cahced eigen values (K = USU^T)
	//Chol
	MatrixXdChol cholKCache; //!< cached cholesky representation of K
	bool KCacheNull,SVDCacheNull,cholKCacheNull;
	void updateSVD();
	void validateCache();
public:
	CCovarianceFunctionCacheOld()
	{
	}

	CCovarianceFunctionCacheOld(PCovarianceFunction covar)
	{
		setCovar(covar);
	}

	virtual ~CCovarianceFunctionCacheOld()
	{
	};

	void setCovar(PCovarianceFunction covar);
	//transmit add and delete sync child to covar object
	virtual void addSyncChild(Pbool l)
	{
		covar->addSyncChild(l);
	}
	virtual void delSyncChild(Pbool l)
	{
		covar->delSyncChild(l);
	}

	//1. getter/setter
	inline PCovarianceFunction getCovar()
	{ return covar;}
	//2. cache interface
	virtual MatrixXd& rgetK();
	virtual MatrixXd& rgetUK();
	virtual VectorXd& rgetSK();
	virtual MatrixXdChol& rgetCholK();
};
typedef sptr<CCovarianceFunctionCacheOld> PCovarianceFunctionCacheOld;






/*Inline functions*/


inline MatrixXd ACovarianceFunction::getX() const
{
	MatrixXd rv;
	this->agetX(&rv);
	return rv;
}

inline  MatrixXd ACovarianceFunction::K() const
{
	MatrixXd RV;
	aK(&RV);
	return RV;
}

inline  VectorXd ACovarianceFunction::Kdiag() const
{
	VectorXd RV;
	aKdiag(&RV);
	return RV;
}

inline  MatrixXd ACovarianceFunction::Kcross( const CovarInput& Xstar ) const 
{
	MatrixXd RV;
	aKcross(&RV,Xstar);
	return RV;
}

inline VectorXd ACovarianceFunction::Kcross_diag(const CovarInput& Xstar) const 
{
	VectorXd RV;
	aKcross_diag(&RV,Xstar);
	return RV;
}

inline MatrixXd ACovarianceFunction::Kgrad_param(const muint_t i) const 
{
	MatrixXd RV;
	aKgrad_param(&RV,i);
	return RV;
}
    
inline MatrixXd ACovarianceFunction::Khess_param(const muint_t i, const muint_t j) const 
{
    MatrixXd RV;
    aKhess_param(&RV,i,j);
    return RV;
}

inline MatrixXd ACovarianceFunction::Kcross_grad_X(const CovarInput & Xstar, const muint_t d) const 
{
	MatrixXd RV;
	aKcross_grad_X(&RV,Xstar,d);
	return RV;
}

inline MatrixXd ACovarianceFunction::Kgrad_X(const muint_t d) const 
{
	MatrixXd RV;
	aKgrad_X(&RV,d);
	return RV;
}


inline VectorXd ACovarianceFunction::Kdiag_grad_X(const muint_t d) const 
{
	VectorXd RV;
	aKdiag_grad_X(&RV,d);
	return RV;
}




inline void ACovarianceFunction::checkWithinDimensions(muint_t d) const 
{
	if (d>=getNumberDimensions())
	{
		std::ostringstream os;
		os << "Dimension index ("<<d<<") out of range in covariance (0.."<<getNumberDimensions()<<").";
		throw CLimixException(os.str());
	}
}
inline void ACovarianceFunction::checkWithinParams(muint_t i) const 
{
	if (i>=getNumberParams())
	{
		std::ostringstream os;
		os << "Parameter index ("<<i<<") out of range in covariance (0.."<<getNumberParams()<<").";
		throw CLimixException(os.str());
	}
}


inline void ACovarianceFunction::checkXDimensions(const CovarInput& X) const 
{
	if ((muint_t)X.cols()!=this->getNumberDimensions())
	{
		std::ostringstream os;
		os << "X("<<(muint_t)X.rows()<<","<<(muint_t)X.cols()<<") column dimension missmatch (covariance: "<<this->getNumberDimensions() <<")";
		throw CLimixException(os.str());
	}
}
inline void ACovarianceFunction::checkParamDimensions(const CovarParams& params) const 
{
	if ((muint_t)(params.rows()) != this->getNumberParams()){
		std::ostringstream os;
		os << "Wrong number of params for covariance funtion " << this->getName() << ". numberParams = " << this->getNumberParams() << ", params.cols() = " << params.cols();
		throw CLimixException(os.str());
	}
}

} /* namespace limix */


#endif /* ACOVARIANCE_H_ */
