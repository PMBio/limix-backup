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
string printDim(const Eigen::EigenBase<Derived>& m)
{
	ostringstream os;
	os << "("<<m.rows() << "," << m.cols()<<")";
	return os.str();
}

//calculate log determinant form cholesky factor
mfloat_t logdet(Eigen::LLT<gpmix::MatrixXd>& chol);
mfloat_t logdet(Eigen::LDLT<gpmix::MatrixXd>& chol);


template <typename Derived>
void arrayInverseInplace(const Eigen::EigenBase<Derived>& m_)
{
	Eigen::EigenBase<Derived>& m = const_cast< Eigen::EigenBase<Derived>& >(m_);

	for (muint_t r=0;r<(muint_t)m.rows();++r)
		for(muint_t c=0;c<(muint_t)m.cols();++c)
			m(r,c) = 1.0/m(r,c);
}

inline void logInplace(MatrixXd& m)
{
	for (muint_t r=0;r<(muint_t)m.rows();++r)
		for(muint_t c=0;c<(muint_t)m.cols();++c)
			m(r,c) = log(m(r,c));
}



MatrixXd randn(const muint_t n, const muint_t m);
MatrixXd Mrand(const muint_t n,const muint_t m);


}
#endif /* MATRIX_HELPER_H_ */
