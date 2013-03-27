/*
 * freeform.h
 *
 *  Created on: Jan 16, 2012
 *      Author: stegle
 */

#ifndef FREEFORM_H_
#define FREEFORM_H_

#include "covariance.h"

namespace limix {

#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
%ignore CFreeFormCF::getIparamDiag;
%ignore CFreeFormCF::K0Covar2Params;
%rename(getIparamDiag) CFreeFormCF::agetIparamDiag;
%rename(getK0) CFreeFormCF::agetK0;
#endif

enum CFreeFromCFConstraitType {freeform,diagonal,dense};

 class CFreeFormCF: public ACovarianceFunction {
protected:
	muint_t numberGroups;
	void agetK0grad_param(MatrixXd* out,muint_t i) const throw(CGPMixException);
	void projectKcross(MatrixXd* out,const MatrixXd& K0,const CovarInput& Xstar) const throw (CGPMixException);
	static muint_t calcNumberParams(muint_t numberGroups);

	CFreeFromCFConstraitType constraint;
	//helper function to convert from matrix to hyperparams

	void aK0Covar2Params(VectorXd* out,const MatrixXd& K0);
	virtual void agetL0grad_param(MatrixXd* out,muint_t i) const throw(CGPMixException);
	virtual void agetL0grad_param_dense(MatrixXd* out,muint_t i) const throw(CGPMixException);

public:

	CFreeFormCF(muint_t numberGroups,CFreeFromCFConstraitType constraint=freeform);
	virtual ~CFreeFormCF();

	virtual void agetL0(MatrixXd* out) const;
	virtual void agetL0_dense(MatrixXd* out) const;
	virtual void agetK0(MatrixXd* out) const;


	void aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException);
	void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException);

	void aKgrad_param(MatrixXd* out,const muint_t i) const throw(CGPMixException);
    void aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const throw(CGPMixException);
	void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const throw(CGPMixException);
	void aKdiag_grad_X(VectorXd* out,const muint_t d) const throw(CGPMixException);

	virtual void agetParamBounds0(CovarParams* lower,CovarParams* upper) const;
	virtual void agetParamMask0(CovarParams* out) const;

	//class information
	inline std::string getName() const {return "CFreeform";};

	//set the a covariance matrix rather than parameters:
	virtual void setParamsCovariance(const MatrixXd& K0) throw(CGPMixException);

	//information on parameter settings
	void agetIparamDiag(VectorXi* out) const;
	VectorXi getIparamDiag() const
	{
		VectorXi rv;
		agetIparamDiag(&rv);
		return rv;
	}

	CFreeFromCFConstraitType getConstraint() const {
		return constraint;
	}

	void setConstraint(CFreeFromCFConstraitType constraint) {
		this->constraint = constraint;
	}
};
typedef sptr<CFreeFormCF> PFreeFormCF;

    
    
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//ignore C++ versions
%ignore CTraitCF::getK0;
%ignore CTraitCF::getK0grad_param;
//rename argout versions for python; this overwrites the C++ convenience functions
%rename(getK0) CTraitCF::agetK0;
%rename(getK0grad_param) CTraitCF::agetK0grad_param;
%rename(getK0hess_param) CTraitCF::agetK0hess_param;
#endif
    
class CTraitCF: public ACovarianceFunction {
        
protected:
    muint_t numberGroups;
    void projectKcross(MatrixXd* out,const MatrixXd& K0,const CovarInput& Xstar) const throw (CGPMixException);
        
public:
        
    CTraitCF(muint_t numberGroups);
    virtual ~CTraitCF();
        
    //pure functions of the TraitCF class that need to be implemented
    virtual void agetK0(MatrixXd* out) const throw(CGPMixException) = 0;
    virtual void agetK0grad_param(MatrixXd* out,muint_t i) const throw(CGPMixException) = 0;
    virtual void agetK0hess_param(MatrixXd* out,muint_t i,muint_t j) const throw(CGPMixException) = 0;
    virtual void setParamsCovariance(const MatrixXd& K0) throw(CGPMixException) = 0;
        
    //Inline convenience functions
    inline MatrixXd getK0() const throw(CGPMixException);
    inline MatrixXd getK0grad_param(muint_t i) const throw(CGPMixException);
        
    //implemented functions
    virtual muint_t getNumberGroups() const;
    
    void aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException);
    void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException);
    
    void aKgrad_param(MatrixXd* out,const muint_t i) const throw(CGPMixException);
    void aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const throw(CGPMixException);
    void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const throw(CGPMixException);
    void aKdiag_grad_X(VectorXd* out,const muint_t d) const throw(CGPMixException);
        
};
typedef sptr<CTraitCF> PTraitCF;

    
inline MatrixXd CTraitCF::getK0() const throw(CGPMixException)
{
    MatrixXd RV;
    agetK0(&RV);
    return RV;
}
    
inline MatrixXd CTraitCF::getK0grad_param(muint_t i) const throw(CGPMixException)
{
    MatrixXd RV;
    agetK0grad_param(&RV,i);
    return RV;
}
    
    
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//ignore C++ versions
%ignore CTFreeFormCF::getIparamDiag;
%ignore CTFreeFormCF::K0Covar2Params;
//rename argout versions for python; this overwrites the C++ convenience functions
%rename(getIparamDiag) CTFreeFormCF::agetIparamDiag;
#endif
    
class CTFreeFormCF: public CTraitCF {
        
protected:
        
    //Calculate the number of parameter: should not it be virtual?
    static muint_t calcNumberParams(muint_t numberGroups);
    
    //helper function to convert from matrix to hyperparams
    void aK0Covar2Params(VectorXd* out,const MatrixXd& K0);
        
public:
    
    CTFreeFormCF(muint_t numberGroups);
    ~CTFreeFormCF();
    
    //TraitCF pure functions
    virtual void agetK0(MatrixXd* out) const throw(CGPMixException);
    virtual void agetK0grad_param(MatrixXd* out,muint_t i) const throw(CGPMixException);
    virtual void agetK0hess_param(MatrixXd* out,muint_t i,muint_t j) const throw(CGPMixException);
    virtual void setParamsCovariance(const MatrixXd& K0) throw(CGPMixException);
    
    //Covariance pure functions
    virtual void agetParamBounds0(CovarParams* lower,CovarParams* upper) const;
    virtual void agetParamMask0(CovarParams* out) const;
    
    //class information
    inline std::string getName() const {return "CTFreeformCF";};
    
    //FreeForm-specific functions
    virtual void setParamsVarCorr(const CovarParams& paramsVC) throw(CGPMixException);
    virtual void agetL0(MatrixXd* out) const;
    virtual void agetL0grad_param(MatrixXd* out,muint_t i) const throw(CGPMixException);
    //information on parameter settings
    void agetIparamDiag(VectorXi* out) const;
    VectorXi getIparamDiag() const
    {
        VectorXi rv;
        agetIparamDiag(&rv);
        return rv;
    }
};
typedef sptr<CTFreeFormCF> PTFreeFormCF;
    
    
    
class CTDenseCF: public CTraitCF {
        
protected:
        
public:
        
    CTDenseCF(muint_t numberGroups);
    ~CTDenseCF();

    //TraitCF pure functions
    virtual void agetK0(MatrixXd* out) const throw(CGPMixException);
    virtual void agetK0grad_param(MatrixXd* out,muint_t i) const throw(CGPMixException);
    virtual void agetK0hess_param(MatrixXd* out,muint_t i,muint_t j) const throw(CGPMixException);
    virtual void setParamsCovariance(const MatrixXd& K0) throw(CGPMixException);
        
    //Covariance pure functions
    virtual void agetParamBounds0(CovarParams* lower,CovarParams* upper) const;
    virtual void agetParamMask0(CovarParams* out) const;
        
    //class information
    inline std::string getName() const {return "CTDenseCF";};
        
};
typedef sptr<CTDenseCF> PTDenseCF;
    
    
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
%rename(getK0) CTFixedCF::agetK0;
#endif
    
    
class CTFixedCF: public CTraitCF {
        
protected:
    MatrixXd K0;
        
public:
    
    CTFixedCF(muint_t numberGroups, const MatrixXd& K0);
    ~CTFixedCF();
        
    //TraitCF pure functions
    virtual void agetK0(MatrixXd* out) const throw(CGPMixException);
    virtual void agetK0grad_param(MatrixXd* out,muint_t i) const throw(CGPMixException);
    virtual void agetK0hess_param(MatrixXd* out,muint_t i,muint_t j) const throw(CGPMixException);
    virtual void setParamsCovariance(const MatrixXd& K0) throw(CGPMixException);
        
    //Covariance pure functions
    virtual void agetParamBounds0(CovarParams* lower,CovarParams* upper) const;
    virtual void agetParamMask0(CovarParams* out) const;
        
    //class information
    inline std::string getName() const {return "CTFixedCF";};
        
};
typedef sptr<CTFixedCF> PTFixedCF;
    
    
    
    
class CTDiagonalCF: public CTraitCF {
        
protected:
        
public:
        
    CTDiagonalCF(muint_t numberGroups);
    ~CTDiagonalCF();

    //ACovarianceCF functions
    virtual void agetParams(CovarParams* out);
    
    //TraitCF pure functions
    virtual void agetK0(MatrixXd* out) const throw(CGPMixException);
    virtual void agetK0grad_param(MatrixXd* out,muint_t i) const throw(CGPMixException);
    virtual void agetK0hess_param(MatrixXd* out,muint_t i,muint_t j) const throw(CGPMixException);
    virtual void setParamsCovariance(const MatrixXd& K0) throw(CGPMixException);
        
    //Covariance pure functions
    virtual void agetParamBounds0(CovarParams* lower,CovarParams* upper) const;
    virtual void agetParamMask0(CovarParams* out) const;
    
    //class information
    inline std::string getName() const {return "CTDiagonalCF";};
        
};
typedef sptr<CTDiagonalCF> PTDiagonalCF;
    

#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
%rename(getK0dense)    CTLowRankCF::agetK0dense;
%rename(getK0diagonal) CTLowRankCF::agetK0diagonal;
#endif
    
    
class CTLowRankCF: public CTraitCF {
        
protected:
        
public:
        
    CTLowRankCF(muint_t numberGroups);
    ~CTLowRankCF();
    
    //LowRank functions
    virtual void agetK0dense(MatrixXd* out) const throw(CGPMixException);
    virtual void agetK0diagonal(MatrixXd* out) const throw(CGPMixException);
    
    //ACovarianceCF functions
    virtual void agetParams(CovarParams* out);
    
    
    //TraitCF pure functions
    virtual void agetK0(MatrixXd* out) const throw(CGPMixException);
    virtual void agetK0grad_param(MatrixXd* out,muint_t i) const throw(CGPMixException);
    virtual void agetK0hess_param(MatrixXd* out,muint_t i,muint_t j) const throw(CGPMixException);
    virtual void setParamsCovariance(const MatrixXd& K0) throw(CGPMixException);

    //Covariance pure functions
    virtual void agetParamBounds0(CovarParams* lower,CovarParams* upper) const;
    virtual void agetParamMask0(CovarParams* out) const;
        
    //class information
    inline std::string getName() const {return "CTLowRankCF";};
        
};
typedef sptr<CTLowRankCF> PTLowRankCF;


    

} /* namespace limix */
#endif /* FREEFORM_H_ */
