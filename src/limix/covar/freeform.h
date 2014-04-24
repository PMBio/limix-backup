// Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
// All rights reserved.
//
// LIMIX is provided under a 2-clause BSD license.
// See license.txt for the complete license.

#ifndef FREEFORM_H_
#define FREEFORM_H_

#include "covariance.h"

namespace limix {

    
/*! \brief Base class for free form covariances that do not depend on the input X
 *
 * Note: as a hack until we implement appropriate handling of non-X covariances, the input dimension is 0
 */
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
	virtual void setX(const CovarInput& X)  {};
	virtual void setXcol(const CovarInput& X, muint_t col)  {};
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const ;
	virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const  {};
	virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const  {};
	//override handling of K dimensions
	virtual muint_t Kdim() const 
		{
			return this->numberGroups;
		}



    virtual void agetScales(CovarParams* out);
    virtual void setParamsCovariance(const MatrixXd& K0) ;

    //Covariance pure functions
	//pure functions that need to be implemented
	virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const ;
	virtual void aKgrad_param(MatrixXd* out,const muint_t i) const ;
    virtual void aKhess_param(MatrixXd* out,const muint_t i,const muint_t j) const ;
    virtual void agetParamMask0(CovarParams* out) const;
    
    //class information
    inline std::string getName() const {return "CFreeFormCF";};
    
    //FreeForm-specific functions
    virtual void setParamsVarCorr(const CovarParams& paramsVC) ;
    virtual void agetL0(MatrixXd* out) const;
    virtual void agetL0grad_param(MatrixXd* out,muint_t i) const ;
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
	virtual void setX(const CovarInput& X)  {};
	virtual void setXcol(const CovarInput& X, muint_t col)  {};
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const ;
	virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const  {};
	virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const  {};
	virtual muint_t Kdim() const 
		{
			return this->numberGroups;
		}


    virtual void agetScales(CovarParams* out);
    virtual void setParamsCovariance(const MatrixXd& K0) ;

    //Covariance pure functions
	//pure functions that need to be implemented
	virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const ;
	virtual void aKgrad_param(MatrixXd* out,const muint_t i) const ;
    virtual void aKhess_param(MatrixXd* out,const muint_t i,const muint_t j) const ;
    virtual void agetParamMask0(CovarParams* out) const;

    //class information
    inline std::string getName() const {return "CRankOneCF";};

};
typedef sptr<CRankOneCF> PRankOneCF;


class CLowRankCF: public ACovarianceFunction {

protected:
    muint_t numberGroups;
    muint_t rank;
public:

    CLowRankCF(muint_t numberGroups, muint_t rank);
    ~CLowRankCF();

    //Block X functions: X is fixed and set in the constructor
    virtual void setX(const CovarInput& X)  {};
    virtual void setXcol(const CovarInput& X, muint_t col)  {};
    virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const ;
    virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const  {};
    virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const  {};
	virtual muint_t Kdim() const 
		{
			return this->numberGroups;
		}


    virtual void agetScales(CovarParams* out);
    virtual void setParamsCovariance(const MatrixXd& K0) ;

    //Covariance pure functions
    //pure functions that need to be implemented
    virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const ;
    virtual void aKgrad_param(MatrixXd* out,const muint_t i) const ;
    virtual void aKhess_param(MatrixXd* out,const muint_t i,const muint_t j) const ;
    virtual void agetParamMask0(CovarParams* out) const;

    //class information
    inline std::string getName() const {return "CLowRankCF";};

};
typedef sptr<CLowRankCF> PLowRankCF;


class CFixedCF: public ACovarianceFunction {

protected:
	muint_t numberGroups;
    MatrixXd K0;
	MatrixXd K0cross;
	VectorXd K0cross_diag;
public:

    CFixedCF(const MatrixXd& K0);
    ~CFixedCF();

    virtual void agetScales(CovarParams* out);
    virtual void setParamsCovariance(const MatrixXd& K0) ;

	//overloaded pure virtual functions:
	virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const ;
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const ;
	virtual void aKgrad_param(MatrixXd* out,const muint_t i) const ;
    virtual void aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const ;
	virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const ;
	virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const ;
	//other overloads
	virtual void aK(MatrixXd* out) const ;

	virtual muint_t Kdim() const 
		{
			return K0.rows();
		}

    virtual void agetParamMask0(CovarParams* out) const;

	//setter and getters
	void setK0(const MatrixXd& K0);
	void setK0cross(const MatrixXd& Kcross);
	void agetK0(MatrixXd* out) const;
	void agetK0cross(MatrixXd* out) const;
	void setK0cross_diag(const VectorXd& Kcross_diag);
	void agetK0cross_diag(VectorXd* out) const;

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
	virtual void setX(const CovarInput& X)  {};
	virtual void setXcol(const CovarInput& X, muint_t col)  {};
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const ;
	virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const  {};
	virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const  {};

    virtual void agetScales(CovarParams* out);
    virtual void setParamsCovariance(const MatrixXd& K0) ;

	virtual muint_t Kdim() const 
		{
			return this->numberGroups;
		}


    //Covariance pure functions
	//pure functions that need to be implemented
	virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const ;
	virtual void aKgrad_param(MatrixXd* out,const muint_t i) const ;
    virtual void aKhess_param(MatrixXd* out,const muint_t i,const muint_t j) const ;
    virtual void agetParamMask0(CovarParams* out) const;

    //class information
    inline std::string getName() const {return "CDiagonalCF";};

};
typedef sptr<CDiagonalCF> PDiagonalCF;
    


class CRank1diagCF: public ACovarianceFunction {

protected:
	muint_t numberGroups;
public:

    CRank1diagCF(muint_t numberGroups);
    ~CRank1diagCF();

	//Block X functions: X is fixed and set in the constructor
	virtual void setX(const CovarInput& X)  {};
	virtual void setXcol(const CovarInput& X, muint_t col)  {};
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const ;
	virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const  {};
	virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const  {};

    virtual void agetScales(CovarParams* out);
    virtual void setParamsCovariance(const MatrixXd& K0) ;

	virtual muint_t Kdim() const 
		{
			return this->numberGroups;
		}


    virtual void agetRank1(MatrixXd* out) const ;
    virtual void agetDiag(MatrixXd* out) const ;
    
    //Covariance pure functions
	//pure functions that need to be implemented
	virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const ;
	virtual void aKgrad_param(MatrixXd* out,const muint_t i) const ;
    virtual void aKhess_param(MatrixXd* out,const muint_t i,const muint_t j) const ;
    virtual void agetParamMask0(CovarParams* out) const;

    //class information
    inline std::string getName() const {return "CRank1diagCF";};

};
typedef sptr<CRank1diagCF> PRank1diagCF;



class CSqExpCF: public ACovarianceFunction {

protected:
    muint_t numberGroups;
    muint_t dim;
public:

    CSqExpCF(muint_t numberGroups, muint_t dim);
    ~CSqExpCF();

    //Block X functions: X is fixed and set in the constructor
    virtual void setX(const CovarInput& X)  {};
    virtual void setXcol(const CovarInput& X, muint_t col)  {};
    virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const ;
    virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const  {};
    virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const  {};

    virtual void agetScales(CovarParams* out);
    virtual void setParamsCovariance(const MatrixXd& K0) ;

	virtual muint_t Kdim() const 
		{
			return this->numberGroups;
		}


    //Covariance pure functions
    //pure functions that need to be implemented
    virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const ;
    virtual void aKgrad_param(MatrixXd* out,const muint_t i) const ;
    virtual void aKhess_param(MatrixXd* out,const muint_t i,const muint_t j) const ;
    virtual void agetParamMask0(CovarParams* out) const;

    //class information
    inline std::string getName() const {return "CSqExpCF";};

};
typedef sptr<CSqExpCF> PSqExpCF;


} /* namespace limix */
#endif /* FREEFORM_H_ */
