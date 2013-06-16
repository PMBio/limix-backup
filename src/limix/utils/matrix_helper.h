/*
 * matrix_helper.h
 *
 *  Created on: Nov 10, 2011
 *      Author: stegle
 */

#ifndef MATRIX_HELPER_H_
#define MATRIX_HELPER_H_

#include <math.h>
#include <cmath>
#include "limix/types.h"

namespace limix{

//create random matrix:
mfloat_t randn(mfloat_t mu=0.0, mfloat_t sigma=1.0);
//helper functions for eigen matrices

template <typename Derived>
bool isnull(const Eigen::EigenBase<Derived>& m)
{
	return (m.size()==0);
}

bool isnull(const Eigen::LLT<MatrixXd>& m);
bool isnull(const Eigen::LDLT<MatrixXd>& m);

template <typename Derived>
std::string printDim(const Eigen::EigenBase<Derived>& m)
{
	std::ostringstream os;
	os << "("<<m.rows() << "," << m.cols()<<")";
	return os.str();
}

//calculate log determinant form cholesky factor
mfloat_t logdet(Eigen::LLT<MatrixXd>& chol);
mfloat_t logdet(Eigen::LDLT<MatrixXd>& chol);


template <typename Derived>
inline void arrayInverseInplace(const Eigen::EigenBase<Derived>& m_)
{
	Eigen::EigenBase<Derived>& m = const_cast< Eigen::EigenBase<Derived>& >(m_);

	for (muint_t r=0;r<(muint_t)m.rows();++r)
		for(muint_t c=0;c<(muint_t)m.cols();++c)
			m(r,c) = 1.0/m(r,c);
}

template <typename Derived>
inline void logInplace(const Eigen::MatrixBase<Derived>& m_)
{
	Eigen::MatrixBase<Derived>& m = const_cast< Eigen::MatrixBase<Derived>& >(m_);

	for (muint_t r=0;r<(muint_t)m.rows();++r)
		for(muint_t c=0;c<(muint_t)m.cols();++c)
			m(r,c) = log(m(r,c));
}

template <typename Derived>
inline void expInplace(const Eigen::MatrixBase<Derived>& m_)
{
	Eigen::MatrixBase<Derived>& m = const_cast< Eigen::MatrixBase<Derived>& >(m_);

	for (muint_t r=0;r<(muint_t)m.rows();++r)
		for(muint_t c=0;c<(muint_t)m.cols();++c)
			m(r,c) = exp(m(r,c));
}


template <typename Derived1,typename Derived2,typename Derived3,typename Derived4>
inline void AexpandMask(const Eigen::MatrixBase<Derived1>& out_,const Eigen::MatrixBase<Derived2>& m,const Eigen::MatrixBase<Derived3>& filter_row, const Eigen::MatrixBase<Derived4>& filter_col) throw (CGPMixException)
{
	Eigen::MatrixBase<Derived1>& out = const_cast< Eigen::MatrixBase<Derived1>& >(out_);
	//0. check consistencey
	if ((filter_row.count()!=m.rows()) || (filter_col.count()!=m.cols()))
	{
		throw CGPMixException("expandMask: filter and array inconsistent");
	}

	//1. expand out array
	out.derived().resize(filter_row.rows(),filter_col.rows());
	muint_t ior=0;
	muint_t ioc=0;
	for (muint_t ir=0;ir<(muint_t)filter_row.rows();++ir)
	{
		if (filter_row(ir))
		{
			ioc=0;
			for(muint_t ic=0;ic<(muint_t)filter_col.rows();++ic)
			{
				out(ir,ic) = m(ior,ioc);
				ioc++;
			}
			ior++;
		}
	}
}

template <typename Derived1,typename Derived3,typename Derived4>
inline void AfilterMask(const Eigen::MatrixBase<Derived1>& out_,const Eigen::MatrixBase<Derived1>& m,const Eigen::MatrixBase<Derived3>& filter_row, const Eigen::MatrixBase<Derived4>& filter_col) throw (CGPMixException)
{
	Eigen::MatrixBase<Derived1>& out = const_cast< Eigen::MatrixBase<Derived1>& >(out_);
	if ((filter_row.rows()!=m.rows()) || (filter_col.rows()!=m.cols()))
	{
		throw CGPMixException("expandMask: filter and array inconsistent");
	}
	//1. reshape result matrix according to filter:
	muint_t num_rows = filter_row.count();
	muint_t num_cols = filter_col.count();
	out.derived().resize(num_rows,num_cols);

	//2. copy
	//index on out array
	muint_t ior=0;
	muint_t ioc=0;
	for (muint_t ir=0;ir<(muint_t)m.rows();++ir)
	{
		if (filter_row(ir))
		{
			ioc=0;
			for(muint_t ic=0;ic<(muint_t)m.cols();++ic)
			{
				out(ior,ioc) = m(ir,ic);
				//out.block(ior,ioc,1,1) = m.block(ir,ic,1,1);
				//out(0,0) = 2;
				//out(ior,ioc) = 1;
				//out(ior,ioc) = m(ir,ic);
				ioc++;
			}
			ior++;
		}
	}
}




MatrixXd randn(const muint_t n, const muint_t m);
MatrixXd Mrand(const muint_t n,const muint_t m);

/*Inline math functions*/
template <typename Derived1, typename Derived2,typename Derived3,typename Derived4>
inline void akronravel(const Eigen::MatrixBase<Derived1> & out_, const Eigen::MatrixBase<Derived2>& A,const Eigen::MatrixBase<Derived3>& B,const Eigen::MatrixBase<Derived4>& X)
{
	Eigen::MatrixBase<Derived1>& out = const_cast< Eigen::MatrixBase<Derived1>& >(out_);
	out.noalias() = A*X*B.transpose();
}
template <typename Derived1, typename Derived2,typename Derived3,typename Derived4>
inline MatrixXd kronravel(const Eigen::MatrixBase<Derived1> & out_, const Eigen::MatrixBase<Derived2>& A,const Eigen::MatrixBase<Derived3>& B,const Eigen::MatrixBase<Derived4>& X)
{
	MatrixXd out;
	akronravel(out,A,B,X);
	return out;
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
template <typename Derived1, typename Derived2,typename Derived3>
inline MatrixXd krondiag(const Eigen::MatrixBase<Derived2>& v1,const Eigen::MatrixBase<Derived3>& v2)
{
	MatrixXd out;
	akrondiag(out,v1,v2);
	return out;
}


template <typename Derived1, typename Derived2,typename Derived3>
/* Kronecker product
 * out = A kron B
 * if addToOut:
 * out += A kron B
 */
inline void akron(const Eigen::MatrixBase<Derived1> & out_, const Eigen::MatrixBase<Derived2>& v1,const Eigen::MatrixBase<Derived3>& v2,bool addToOut=false)
{
	Eigen::MatrixBase<Derived1>& out = const_cast< Eigen::MatrixBase<Derived1>& >(out_);
	out.derived().resize(v1.rows()*v2.rows(),v1.cols()*v2.cols());
	for (muint_t ir=0;ir<(muint_t)v1.rows();++ir)
		for (muint_t ic=0;ic<(muint_t)v1.cols();++ic)
		{
			if (addToOut)
				out.block(ir*v2.rows(),ic*v2.cols(),v2.rows(),v2.cols()) += v1(ir,ic)*v2;
			else
				out.block(ir*v2.rows(),ic*v2.cols(),v2.rows(),v2.cols()) = v1(ir,ic)*v2;
		}
}

template <typename Derived1, typename Derived2,typename Derived3>
inline MatrixXd kron(const Eigen::MatrixBase<Derived2>& v1,const Eigen::MatrixBase<Derived3>& v2)
{
	MatrixXd out;
	akron(out,v1,2);
	return out;
}



template <typename Derived1, typename Derived2,typename Derived3>
inline void akron_diag(const Eigen::MatrixBase<Derived1> & out_, const Eigen::MatrixBase<Derived2>& v1,const Eigen::MatrixBase<Derived3>& v2,bool addToOut=false)
{
	Eigen::MatrixBase<Derived1>& out = const_cast< Eigen::MatrixBase<Derived1>& >(out_);
	out.derived().resize(v1.rows()*v2.rows(),1);
	for (muint_t ir=0;ir<(muint_t)v1.rows();++ir)
		out.block(ir*v2.rows(),0,v2.rows(),1) = v1(ir,ir)*v2.diagonal();
}

template <typename Derived1, typename Derived2,typename Derived3>
inline MatrixXd kron_diag(const Eigen::MatrixBase<Derived2>& v1,const Eigen::MatrixBase<Derived3>& v2)
{
	MatrixXd out;
	akron_diag(out,v1,2);
	return out;
}


template <typename Derived1>
inline mfloat_t getVarianceK(const Eigen::MatrixBase<Derived1> & K_) throw(CGPMixException)
{
	//cast out arguments
	//Eigen::MatrixBase<Derived1>& K = const_cast< Eigen::MatrixBase<Derived1>& >(K);
	//ensure that it is a square matrix:
	if (K_.rows()!=K_.cols())
		throw CGPMixException("Kernel scaling requires square kernel matrix");

	//diagonal
	mfloat_t c = K_.trace();
	c -= 1.0/K_.rows() * K_.sum();

	mfloat_t scalar = c/(K_.rows()-1);
	return scalar;
}


template <typename Derived1>
inline void VarianceScaleK(const Eigen::MatrixBase<Derived1> & K_) throw(CGPMixException)
{
	//cast out arguments
	Eigen::MatrixBase<Derived1>& K = const_cast< Eigen::MatrixBase<Derived1>& >(K);

	mfloat_t scalar = getVarianceK(K);
	K/=scalar;
}



}
#endif /* MATRIX_HELPER_H_ */
