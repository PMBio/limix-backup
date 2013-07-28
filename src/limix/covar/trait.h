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
%ignore CTrait::getK0;
%ignore CTrait::getK0grad_param;
//rename argout versions for python; this overwrites the C++ convenience functions
%rename(getK0) CTrait::agetK0;
%rename(getK0grad_param) CTrait::agetK0grad_param;
%rename(getK0hess_param) CTrait::agetK0hess_param;
#endif
    
class CTrait: public ACovarianceFunction {
protected:
    muint_t numberGroups;
public:
    CTrait(muint_t numberGroups);
    virtual ~CTrait();

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
typedef sptr<CTrait> PTrait;
    
    
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//ignore C++ versions
%ignore CTFreeForm::getIparamDiag;
%ignore CTFreeForm::K0Covar2Params;
//rename argout versions for python; this overwrites the C++ convenience functions
%rename(getIparamDiag) CTFreeForm::agetIparamDiag;
#endif
    
class CTFreeForm: public CTrait {
        
protected:
    //Calculate the number of parameter: should not it be virtual?
    static muint_t calcNumberParams(muint_t numberGroups);
    //helper function to convert from matrix to hyperparams
    void aK0Covar2Params(VectorXd* out,const MatrixXd& K0);
public:
    
    CTFreeForm(muint_t numberGroups);
    ~CTFreeForm();
    
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
    inline std::string getName() const {return "CTFreeForm";};
    
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
typedef sptr<CTFreeForm> PTFreeForm;
    

class CTDense: public CTrait {
        
protected:
        
public:
        
    CTDense(muint_t numberGroups);
    ~CTDense();

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
    inline std::string getName() const {return "CTDense";};
        
};
typedef sptr<CTDense> PTDense;
    
    
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
%rename(getK0) CTFixed::agetK0;
#endif
    


class CTFixed: public CTrait {
        
protected:
    MatrixXd K0;
        
public:
    
    CTFixed(muint_t numberGroups, const MatrixXd& K0);
    ~CTFixed();
    
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
    inline std::string getName() const {return "CTFixed";};
        
};
typedef sptr<CTFixed> PTFixed;
    
    
class CTDiagonal: public CTrait {
        
protected:
        
public:
        
    CTDiagonal(muint_t numberGroups);
    ~CTDiagonal();
    
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
    inline std::string getName() const {return "CTDiagonal";};
        
};
typedef sptr<CTDiagonal> PTDiagonal;
    

#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
%rename(getK0dense)    CTLowRank::agetK0dense;
%rename(getK0diagonal) CTLowRank::agetK0diagonal;
#endif
    
    
class CTLowRank: public CTrait {
        
protected:

public:
        
    CTLowRank(muint_t numberGroups);
    ~CTLowRank();
    
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
    inline std::string getName() const {return "CTLowRank";};
        
};
typedef sptr<CTLowRank> PTLowRank;

} /* namespace limix */
#endif /* TRAIT_H_ */
