/*
 * gp_kronecker.h
 *
 *  Created on: Jan 2, 2012
 *      Author: stegle
 */

#ifndef GP_KRONECKER_H_
#define GP_KRONECKER_H_

#include "gp_base.h"

namespace gpmix {

/*Inline math functions*/
template <typename Derived1, typename Derived2,typename Derived3,typename Derived4>
inline void akronravel(const Eigen::MatrixBase<Derived1> & out_, const Eigen::MatrixBase<Derived2>& A,const Eigen::MatrixBase<Derived3>& B,const Eigen::MatrixBase<Derived4>& X)
{
	Eigen::MatrixBase<Derived1>& out = const_cast< Eigen::MatrixBase<Derived1>& >(out_);
	out.noalias() = A*X*B.transpose();
}

template <typename Derived1, typename Derived2,typename Derived3>
inline void akrondiag(const Eigen::MatrixBase<Derived1> & out_, const Eigen::MatrixBase<Derived2>& v1,const Eigen::MatrixBase<Derived3>& v2)
{
	Eigen::MatrixBase<Derived1>& out = const_cast< Eigen::MatrixBase<Derived1>& >(out_);
	out.derived().resize(v1.rows(),v2.rows());
	out.rowwise()  = v2.transpose();
	//loop and multiply v1
	for (muint_t ic=0;ic<(muint_t)out.cols();ic++)
		out.col(ic).array() *= v1.array();
}


//TODO: remove test script for shared pointers
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
#endif
class CTest
{
public:
	CTest(muint_t i)
	{
		std::cout << i << "\n";
	};
	virtual ~CTest()
	{};
	void print0(CTest& other)
	{
			std::cout << "prin called" << "\n";
	}
	void print1(sptr<CTest> other)
	{
		std::cout << "prin called" << "\n";
	}
};
//typedef sptr<gpmix::CTest> PTest;

//forward definition:
class CGPkronecker;
class CGPKroneckerCache;

//caching module for SVD
class CGPSVDCache : public CGPCholCache
{
	friend class CGPKroneckerCache;
protected:
	MatrixXd UK;
	VectorXd SK;
	bool USVDNULL;
	PCovarianceFunction covar;
	void updateDecomposition();
public:
	CGPSVDCache(CGPbase* gp, PCovarianceFunction covar);
	virtual ~CGPSVDCache()
	{};
	virtual void clearCache();
	virtual bool isInSync() const;

	MatrixXd& getUK();
	VectorXd& getSK();
    ACovarianceFunction& getCovar();
    void agetUK(MatrixXd* out)
    {
    	(*out) = getUK();
    }
    void agetSK(VectorXd* out)
    {
    	(*out) = getSK();
    }

};

class CGPKroneckerCache
{
	friend class CGPKronecker;
protected:
	MatrixXd Yrot;
	MatrixXd Si;
	MatrixXd YSi;
	MatrixXd KinvY;
	mfloat_t Knoise;

	CGPkronecker* gp;
	bool YrotNull,SiNull,YSiNull,KinvYNull;
public:
	CGPSVDCache cache_r;
	CGPSVDCache cache_c;

	CGPKroneckerCache(CGPkronecker* gp,PCovarianceFunction covar_r,PCovarianceFunction covar_c );
	virtual ~CGPKroneckerCache()
	{};
	virtual void clearCache();
	virtual bool isInSync() const;
	MatrixXd& getYrot();
	MatrixXd& getSi();
	MatrixXd& getYSi();
	MatrixXd& getKinvY();

	void agetSi(MatrixXd* out)
	{
		(*out) = getSi();
	}
	void agetYSi(MatrixXd* out)
	{
		(*out) = getYSi();
	}
	void agetYrot(MatrixXd* out)
	{
		(*out) = getYrot();
	}
};


#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//%shared_ptr(gpmix::CGPkronecker)
#endif

class CGPkronecker: public CGPbase {
	friend class CGPKroneckerCache;
	virtual void updateParams() throw (CGPMixException);

protected:
	//row and column covariance functions:
	PCovarianceFunction covar_r;
	PCovarianceFunction covar_c;


	//cache:
	CGPKroneckerCache cache;
	VectorXi gplvmDimensions_r;  //gplvm dimensions
	VectorXi gplvmDimensions_c;  //gplvm dimension

	mfloat_t _gradLogDet(MatrixXd& dK,bool columns);
	mfloat_t _gradQuadrForm(MatrixXd& dK,bool columns);
	void _gradQuadrFormX(VectorXd* rv,MatrixXd& dK,bool columns);
	void _gradLogDetX(VectorXd* out, MatrixXd& dK,bool columns);

public:
	CGPkronecker(PCovarianceFunction covar_r, PCovarianceFunction covar_c, PLikelihood lik=PLikelihood(),PDataTerm mean=PDataTerm());
	virtual ~CGPkronecker();

	void setX_r(const CovarInput& X) throw (CGPMixException);
	void setX_c(const CovarInput& X) throw (CGPMixException);
	void setY(const MatrixXd& Y);
	void setCovar_r(PCovarianceFunction covar);
	void setCovar_c(PCovarianceFunction covar);


	mfloat_t LML() throw (CGPMixException);
	virtual mfloat_t LML(const CGPHyperParams& params) throw (CGPMixException)
	{
		return CGPbase::LML(params);
	}
	//same for concatenated list of parameters
	virtual mfloat_t LML(const VectorXd& params) throw (CGPMixException)
	{
		return CGPbase::LML(params);
	}

	PLikNormalSVD getLik()
	{
		PLikNormalSVD RV = std::tr1::static_pointer_cast<CLikNormalSVD> (this->lik);
		return RV;
	}


	CGPHyperParams LMLgrad() throw (CGPMixException);
	virtual void aLMLgrad_covar(VectorXd* out,bool columns) throw (CGPMixException);
	virtual void aLMLgrad_covar_r(VectorXd* out) throw (CGPMixException);
	virtual void aLMLgrad_covar_c(VectorXd* out) throw (CGPMixException);
	virtual void aLMLgrad_lik(VectorXd* out) throw (CGPMixException);
	virtual void aLMLgrad_X_r(MatrixXd* out) throw (CGPMixException);
	virtual void aLMLgrad_X_c(MatrixXd* out) throw (CGPMixException);
	virtual void aLMLgrad_dataTerm(MatrixXd* out) throw (CGPMixException);
    CGPKroneckerCache& getCache();
    PCovarianceFunction  getCovarC() const;
    PCovarianceFunction  getCovarR() const;
    VectorXi getGplvmDimensionsC() const;
    VectorXi getGplvmDimensionsR() const;
    void setGplvmDimensionsC(VectorXi gplvmDimensionsC);
    void setGplvmDimensionsR(VectorXi gplvmDimensionsR);
};
typedef sptr<CGPkronecker> PGPkronecker;


} /* namespace gpmix */
#endif /* GP_KRONECKER_H_ */
