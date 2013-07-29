/*
 * trait.h
 *
 *  Created on: Jan 16, 2012
 *      Author: stegle
 */

#ifndef FREEFORM_H_
#define FREEFORM_H_

#include "covariance.h"

namespace limix {

    
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//ignore C++ versions
%ignore CFreeFormCF::getIparamDiag;
%ignore CFreeFormCF::K0Covar2Params;
//rename argout versions for python; this overwrites the C++ convenience functions
%rename(getIparamDiag) CFreeFormCF::agetIparamDiag;
#endif

class CFreeFormCF: public ACovarianceFunction {
        
protected:
	muint_t numberGroups;
    //Calculate the number of parameter
    static muint_t calcNumberParams(muint_t numberGroups);
    //helper function to convert from matrix to hyperparams
    void aK0Covar2Params(VectorXd* out,const MatrixXd& K0);
public:
    
    CFreeFormCF(muint_t numberGroups);
    ~CFreeFormCF();
    
	//Block X functions: X is fixed and set in the constructor
	virtual void setX(const CovarInput& X) throw (CGPMixException) {};
	virtual void setXcol(const CovarInput& X, muint_t col) throw (CGPMixException) {};
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException);
	virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const throw(CGPMixException) {};
	virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const throw(CGPMixException) {};

    virtual void agetScales(CovarParams* out);
    virtual void setParamsCovariance(const MatrixXd& K0) throw(CGPMixException);

    //Covariance pure functions
	//pure functions that need to be implemented
	virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException);
	virtual void aKgrad_param(MatrixXd* out,const muint_t i) const throw(CGPMixException);
    virtual void aKhess_param(MatrixXd* out,const muint_t i,const muint_t j) const throw(CGPMixException);
    virtual void agetParamMask0(CovarParams* out) const;
    
    //class information
    inline std::string getName() const {return "CFreeFormCF";};
    
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
typedef sptr<CFreeFormCF> PFreeFormCF;



class CRankOneCF: public ACovarianceFunction {
        
protected:
    muint_t numberGroups;
public:
        
    CRankOneCF(muint_t numberGroups);
    ~CRankOneCF();

	//Block X functions: X is fixed and set in the constructor
	virtual void setX(const CovarInput& X) throw (CGPMixException) {};
	virtual void setXcol(const CovarInput& X, muint_t col) throw (CGPMixException) {};
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException);
	virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const throw(CGPMixException) {};
	virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const throw(CGPMixException) {};

    virtual void agetScales(CovarParams* out);
    virtual void setParamsCovariance(const MatrixXd& K0) throw(CGPMixException);
        
    //Covariance pure functions
    virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException);
    virtual void aKgrad_param(MatrixXd* out,muint_t i) const throw(CGPMixException);
    virtual void aKhess_param(MatrixXd* out,muint_t i,muint_t j) const throw(CGPMixException);
    virtual void agetParamMask0(CovarParams* out) const;
        
    //class information
    inline std::string getName() const {return "CRankOneCF";};
        
};
typedef sptr<CRankOneCF> PRankOneCF;
    
    
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
%rename(getK0) CFixedCF::agetK0;
#endif
    


class CFixedCF: public ACovarianceFunction {
        
protected:
	muint_t numberGroups;
    MatrixXd K0;
        
public:
    CFixedCF(const MatrixXd& K0);
    ~CFixedCF();
    
	//Block X functions: X is fixed and set in the constructor
	virtual void setX(const CovarInput& X) throw (CGPMixException) {};
	virtual void setXcol(const CovarInput& X, muint_t col) throw (CGPMixException) {};
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException);
	virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const throw(CGPMixException) {};
	virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const throw(CGPMixException) {};

    virtual void agetScales(CovarParams* out);
    virtual void setParamsCovariance(const MatrixXd& K0) throw(CGPMixException);

    //Covariance pure functions
    virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException);
    virtual void aKgrad_param(MatrixXd* out,muint_t i) const throw(CGPMixException);
    virtual void aKhess_param(MatrixXd* out,muint_t i,muint_t j) const throw(CGPMixException);
    virtual void agetParamMask0(CovarParams* out) const;
        
    //class information
    inline std::string getName() const {return "CFixedCF";};
        
};
typedef sptr<CFixedCF> PFixedCF;
    
    
class CDiagonalCF: public ACovarianceFunction {
        
protected:
	muint_t numberGroups;
public:
        
    CDiagonalCF(muint_t numberGroups);
    ~CDiagonalCF();
    
	//Block X functions: X is fixed and set in the constructor
	virtual void setX(const CovarInput& X) throw (CGPMixException) {};
	virtual void setXcol(const CovarInput& X, muint_t col) throw (CGPMixException) {};
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException);
	virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const throw(CGPMixException) {};
	virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const throw(CGPMixException) {};

    virtual void agetScales(CovarParams* out);
    virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException);
    virtual void aKgrad_param(MatrixXd* out,muint_t i) const throw(CGPMixException);
    virtual void aKhess_param(MatrixXd* out,muint_t i,muint_t j) const throw(CGPMixException);
    virtual void setParamsCovariance(const MatrixXd& K0) throw(CGPMixException);
        
    //Covariance pure functions
    virtual void agetParamMask0(CovarParams* out) const;
    
    //class information
    inline std::string getName() const {return "CDiagonalCF";};
        
};
typedef sptr<CDiagonalCF> PDiagonalCF;
    

#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
%rename(getK0dense)    CLowRankCF::agetK0dense;
%rename(getK0diagonal) CLowRankCF::agetK0diagonal;
#endif
    
    
class CLowRankCF: public ACovarianceFunction {
        
protected:
	muint_t numberGroups;
public:
        
    CLowRankCF(muint_t numberGroups);
    ~CLowRankCF();
    
	//Block X functions: X is fixed and set in the constructor
	virtual void setX(const CovarInput& X) throw (CGPMixException) {};
	virtual void setXcol(const CovarInput& X, muint_t col) throw (CGPMixException) {};
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException);
	virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const throw(CGPMixException) {};
	virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const throw(CGPMixException) {};

    //LowRank functions
    virtual void agetK0dense(MatrixXd* out) const throw(CGPMixException);
    virtual void agetK0diagonal(MatrixXd* out) const throw(CGPMixException);
    
    virtual void agetScales(CovarParams* out);
    virtual void setParamsCovariance(const MatrixXd& K0) throw(CGPMixException);

    //Covariance pure functions
    virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException);
    virtual void aKgrad_param(MatrixXd* out,muint_t i) const throw(CGPMixException);
    virtual void aKhess_param(MatrixXd* out,muint_t i,muint_t j) const throw(CGPMixException);
    virtual void agetParamMask0(CovarParams* out) const;
        
    //class information
    inline std::string getName() const {return "CLowRankCF";};
        
};
typedef sptr<CLowRankCF> PLowRankCF;

} /* namespace limix */
#endif /* TRAIT_H_ */
