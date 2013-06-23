/*
 * trait.h
 *
 *  Created on: Jan 16, 2012
 *      Author: stegle
 */

#ifndef TRAIT_H_
#define TRAIT_H_

#include "covariance.h"

namespace limix {
    
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//ignore C++ versions
%ignore CTraitNew::getK0;
%ignore CTraitNew::getK0grad_param;
//rename argout versions for python; this overwrites the C++ convenience functions
%rename(getK0) CTraitNew::agetK0;
%rename(getK0grad_param) CTraitNew::agetK0grad_param;
%rename(getK0hess_param) CTraitNew::agetK0hess_param;
#endif
    
class CTraitNew: public ACovarianceFunction {
protected:
    muint_t numberGroups;
public:
    CTraitNew(muint_t numberGroups);
    virtual ~CTraitNew();

	//Block X functions: X is fixed and set in the constructor
	virtual void setX(const CovarInput& X) throw (CGPMixException) {};
	virtual void setXcol(const CovarInput& X, muint_t col) throw (CGPMixException) {};
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException);
	virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const throw(CGPMixException) {};
	virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const throw(CGPMixException) {};

    //pure functions of the TraitNew class that need to be implemented
    virtual void agetScales(CovarParams* out) = 0;
    virtual void setParamsCovariance(const MatrixXd& K0) throw(CGPMixException) = 0;
        
    //implemented functions
    virtual muint_t getNumberGroups() const;
};
typedef sptr<CTraitNew> PTraitNew;
    
    
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//ignore C++ versions
%ignore CTFreeFormNew::getIparamDiag;
%ignore CTFreeFormNew::K0Covar2Params;
//rename argout versions for python; this overwrites the C++ convenience functions
%rename(getIparamDiag) CTFreeFormNew::agetIparamDiag;
#endif
    
class CTFreeFormNew: public CTraitNew {
        
protected:
    //Calculate the number of parameter: should not it be virtual?
    static muint_t calcNumberParams(muint_t numberGroups);
    //helper function to convert from matrix to hyperparams
    void aK0Covar2Params(VectorXd* out,const MatrixXd& K0);
public:
    
    CTFreeFormNew(muint_t numberGroups);
    ~CTFreeFormNew();
    
    //TraitNew pure functions
    virtual void agetScales(CovarParams* out);
    virtual void setParamsCovariance(const MatrixXd& K0) throw(CGPMixException);
    
    //Covariance pure functions
	//pure functions that need to be implemented
	virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException);
	virtual void aKgrad_param(MatrixXd* out,const muint_t i) const throw(CGPMixException);
    virtual void aKhess_param(MatrixXd* out,const muint_t i,const muint_t j) const throw(CGPMixException);
    virtual void agetParamBounds0(CovarParams* lower,CovarParams* upper) const;
    virtual void agetParamMask0(CovarParams* out) const;
    
    //class information
    inline std::string getName() const {return "CTFreeFormNew";};
    
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
typedef sptr<CTFreeFormNew> PTFreeFormNew;
    

class CTDenseNew: public CTraitNew {
        
protected:
        
public:
        
    CTDenseNew(muint_t numberGroups);
    ~CTDenseNew();

    //TraitNew pure functions
    virtual void agetScales(CovarParams* out);
    virtual void setParamsCovariance(const MatrixXd& K0) throw(CGPMixException);
        
    //Covariance pure functions
    virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException);
    virtual void aKgrad_param(MatrixXd* out,muint_t i) const throw(CGPMixException);
    virtual void aKhess_param(MatrixXd* out,muint_t i,muint_t j) const throw(CGPMixException);
    virtual void agetParamBounds0(CovarParams* lower,CovarParams* upper) const;
    virtual void agetParamMask0(CovarParams* out) const;
        
    //class information
    inline std::string getName() const {return "CTDenseNew";};
        
};
typedef sptr<CTDenseNew> PTDenseNew;
    
    
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
%rename(getK0) CTFixedNew::agetK0;
#endif
    


class CTFixedNew: public CTraitNew {
        
protected:
    MatrixXd K0;
        
public:
    
    CTFixedNew(muint_t numberGroups, const MatrixXd& K0);
    ~CTFixedNew();
    
    //TraitNew pure functions
    virtual void agetScales(CovarParams* out);
    virtual void setParamsCovariance(const MatrixXd& K0) throw(CGPMixException);

    //Covariance pure functions
    virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException);
    virtual void aKgrad_param(MatrixXd* out,muint_t i) const throw(CGPMixException);
    virtual void aKhess_param(MatrixXd* out,muint_t i,muint_t j) const throw(CGPMixException);
    virtual void agetParamBounds0(CovarParams* lower,CovarParams* upper) const;
    virtual void agetParamMask0(CovarParams* out) const;
        
    //class information
    inline std::string getName() const {return "CTFixedNew";};
        
};
typedef sptr<CTFixedNew> PTFixedNew;
    
    
class CTDiagonalNew: public CTraitNew {
        
protected:
        
public:
        
    CTDiagonalNew(muint_t numberGroups);
    ~CTDiagonalNew();
    
    //TraitNew pure functions
    virtual void agetScales(CovarParams* out);
    virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException);
    virtual void aKgrad_param(MatrixXd* out,muint_t i) const throw(CGPMixException);
    virtual void aKhess_param(MatrixXd* out,muint_t i,muint_t j) const throw(CGPMixException);
    virtual void setParamsCovariance(const MatrixXd& K0) throw(CGPMixException);
        
    //Covariance pure functions
    virtual void agetParamBounds0(CovarParams* lower,CovarParams* upper) const;
    virtual void agetParamMask0(CovarParams* out) const;
    
    //class information
    inline std::string getName() const {return "CTDiagonalNew";};
        
};
typedef sptr<CTDiagonalNew> PTDiagonalNew;
    

#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
%rename(getK0dense)    CTLowRankNew::agetK0dense;
%rename(getK0diagonal) CTLowRankNew::agetK0diagonal;
#endif
    
    
class CTLowRankNew: public CTraitNew {
        
protected:

public:
        
    CTLowRankNew(muint_t numberGroups);
    ~CTLowRankNew();
    
    //LowRank functions
    virtual void agetK0dense(MatrixXd* out) const throw(CGPMixException);
    virtual void agetK0diagonal(MatrixXd* out) const throw(CGPMixException);
    
    //TraitNew pure functions
    virtual void agetScales(CovarParams* out);
    virtual void setParamsCovariance(const MatrixXd& K0) throw(CGPMixException);

    //Covariance pure functions
    virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException);
    virtual void aKgrad_param(MatrixXd* out,muint_t i) const throw(CGPMixException);
    virtual void aKhess_param(MatrixXd* out,muint_t i,muint_t j) const throw(CGPMixException);
    virtual void agetParamBounds0(CovarParams* lower,CovarParams* upper) const;
    virtual void agetParamMask0(CovarParams* out) const;
        
    //class information
    inline std::string getName() const {return "CTLowRankNew";};
        
};
typedef sptr<CTLowRankNew> PTLowRankNew;

} /* namespace limix */
#endif /* TRAIT_H_ */
