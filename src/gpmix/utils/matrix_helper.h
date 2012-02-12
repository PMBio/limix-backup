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
#include <gpmix/types.h>

namespace gpmix{

//create random matrix:
mfloat_t randn(mfloat_t mu=0.0, mfloat_t sigma=1.0);
//helper functions for eigen matrices

template <typename Derived>
bool isnull(const Eigen::EigenBase<Derived>& m)
{
	return (m.size()==0);
}

bool isnull(const Eigen::LLT<gpmix::MatrixXd>& m);
bool isnull(const Eigen::LDLT<gpmix::MatrixXd>& m);

template <typename Derived>
std::string printDim(const Eigen::EigenBase<Derived>& m)
{
	std::ostringstream os;
	os << "("<<m.rows() << "," << m.cols()<<")";
	return os.str();
}

//calculate log determinant form cholesky factor
mfloat_t logdet(Eigen::LLT<gpmix::MatrixXd>& chol);
mfloat_t logdet(Eigen::LDLT<gpmix::MatrixXd>& chol);


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


}
#endif /* MATRIX_HELPER_H_ */
