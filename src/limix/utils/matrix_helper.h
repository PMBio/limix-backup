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


#ifndef MATRIX_HELPER_H_
#define MATRIX_HELPER_H_


#include <math.h>
#include "limix/types.h"

namespace limix{

//create random matrix:
mfloat_t randNormal(mfloat_t mu=0.0, mfloat_t sigma=1.0);
//helper functions for eigen matrices

bool negate(bool in);



template <typename DerivedX>
/*!
 * slice matirx m1 according index Iselect and return to out
 */
void slice(const Eigen::PlainObjectBase<DerivedX> & m1,const MatrixXb& Iselect, Eigen::PlainObjectBase<DerivedX> & out) 
{

	//Eigen::DenseBase<DerivedX>& out = const_cast< Eigen::DenseBase<DerivedX>& >(out_);

	//create large enough temporrary memory
	//Eigen::Matrix<DerivedX,Eigen::Dynamic,Eigen::Dynamic> TMP;
	//TMP.resize(m1.rows(),m1.cols());

	out.resize(m1.rows(),m1.cols());

	//check dimensions of slicing operator
	if (Iselect.cols()==1)
	{
		if (Iselect.rows()!=m1.rows())
			throw CLimixException("Slicing operator needs to be N x 1 or 1 x M if m1 is an N x M matrix");
		//slice along the row dimension
		muint_t rc = 0;
		bool v;
		for(muint_t ir=0;ir<(muint_t)Iselect.rows();++ir)
		{
			v = Iselect(ir,0);
			if(v)
			{
				out.row(rc) = m1.row(ir);
				rc+=1;
			}
		}
		//shrink
		out.conservativeResize(rc,m1.cols());
	}
	else if (Iselect.rows()==1)
	{
		if (Iselect.cols()!=m1.cols())
			throw CLimixException("Slicing operator needs to be N x 1 or 1 x M if m1 is an N x M matrix");
		muint_t cc = 0;
		for (muint_t ic=0;ic<(muint_t)Iselect.cols();++ic)
			if(Iselect(0,ic))
			{
				out.col(cc) = m1.col(ic);
				cc+=1;
			}
		//shrink
		out.conservativeResize(m1.rows(),cc);
	}else
	{
		throw CLimixException("Slicing operator needs to be N x 1 or 1 x M if m1 is an N x M matrix");
	}
}




/*! check individual elements of an eigen matrix for NAN and return a boolean array
 *
 */
MatrixXb isnan(const MatrixXd& m);

bool isnull(const Eigen::LLT<MatrixXd>& m);
bool isnull(const Eigen::LDLT<MatrixXd>& m);
template <typename Derived>
bool isnull(const Eigen::EigenBase<Derived>& m)
{
	return (m.size()==0);
}


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

inline mfloat_t randu()
{
	mfloat_t sample = ((mfloat_t)rand()+0.5);
	mfloat_t max = RAND_MAX+1.0;
	mfloat_t min = 0.0;
	if (sample!=sample || isinf(sample)||sample<=min||sample >=max)
	{
		std::cout <<"nan sample from randn: "<< sample<<"\n";
	}
	sample /= max;
	return sample;
};

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

template <typename Derived>
inline void pow2InPlace(const Eigen::MatrixBase<Derived>& m_)
{
	Eigen::MatrixBase<Derived>& m = const_cast< Eigen::MatrixBase<Derived>& >(m_);

	for (muint_t r=0;r<(muint_t)m.rows();++r)
		for(muint_t c=0;c<(muint_t)m.cols();++c)
			m(r,c) = m(r,c)*m(r,c);
}


template <typename Derived1,typename Derived2,typename Derived3,typename Derived4>
inline void AexpandMask(const Eigen::MatrixBase<Derived1>& out_,const Eigen::MatrixBase<Derived2>& m,const Eigen::MatrixBase<Derived3>& filter_row, const Eigen::MatrixBase<Derived4>& filter_col) 
{
	Eigen::MatrixBase<Derived1>& out = const_cast< Eigen::MatrixBase<Derived1>& >(out_);
	//0. check consistencey
	if ((filter_row.count()!=m.rows()) || (filter_col.count()!=m.cols()))
	{
		throw CLimixException("expandMask: filter and array inconsistent");
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
inline void AfilterMask(const Eigen::MatrixBase<Derived1>& out_,const Eigen::MatrixBase<Derived1>& m,const Eigen::MatrixBase<Derived3>& filter_row, const Eigen::MatrixBase<Derived4>& filter_col) 
{
	Eigen::MatrixBase<Derived1>& out = const_cast< Eigen::MatrixBase<Derived1>& >(out_);
	if ((filter_row.rows()!=m.rows()) || (filter_col.rows()!=m.cols()))
	{
		throw CLimixException("expandMask: filter and array inconsistent");
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


mfloat_t randbeta(mfloat_t a, mfloat_t b);
MatrixXd randbeta(const muint_t n, const muint_t m, mfloat_t a, mfloat_t b);

MatrixXd randn(const muint_t n, const muint_t m);
MatrixXd Mrand(const muint_t n,const muint_t m);
MatrixXd BaldingNichols(muint_t N, muint_t M, mfloat_t mafmin, mfloat_t FST, bool standardize);

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
/* Considers U*S^(alpha) */
inline void aUS2alpha(const Eigen::MatrixBase<Derived1> & out_, const Eigen::MatrixBase<Derived2>& v1,const Eigen::MatrixBase<Derived3>& v2,mfloat_t alpha)
{
	Eigen::MatrixBase<Derived1>& out = const_cast< Eigen::MatrixBase<Derived1>& >(out_);
	out.derived().resize(v1.rows(),v1.cols());
	for (muint_t ir=0;ir<(muint_t)v1.rows();++ir)
		for (muint_t ic=0;ic<(muint_t)v1.cols();++ic)
		{
			out(ir,ic)=v1(ir,ic)*std::pow(v2(ic,0),alpha);
		}
}

template <typename Derived1, typename Derived2,typename Derived3>
/* Considers S^(alpha)*U */
inline void aS2alphaU(const Eigen::MatrixBase<Derived1> & out_, const Eigen::MatrixBase<Derived2>& v1,const Eigen::MatrixBase<Derived3>& v2,mfloat_t alpha)
{
	Eigen::MatrixBase<Derived1>& out = const_cast< Eigen::MatrixBase<Derived1>& >(out_);
	out.derived().resize(v2.rows(),v2.cols());
	for (muint_t ir=0;ir<(muint_t)v1.rows();++ir)
		for (muint_t ic=0;ic<(muint_t)v1.cols();++ic)
		{
			out(ir,ic)=std::pow(v1(ir,0),alpha)*v2(ir,ic);
		}
}


template <typename Derived1, typename Derived2,typename Derived3>

/*! Kronecker product between two matrices A and B
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



/*! Matrix index product
 * out_{i,j} = A_{col(i),col(j)} * B_{row(i),row(j)} where row/col indices as provided in a separate list kroneckerIndicator
 * if addToOut:
 * out += A kron B
 *
 * note: if kroneckerIndicator is null, we will revert to a standard kroneckerproduct!
 * \param v1: column covariance
 * \param v2: row covariance
 */
template <typename Derived1, typename Derived2,typename Derived3,typename Derived4>
inline void aMatrixIndexProduct(const Eigen::MatrixBase<Derived1> & out_, const Eigen::MatrixBase<Derived2>& v1,const Eigen::MatrixBase<Derived3>& v2,const Eigen::MatrixBase<Derived4>& kroneckerIndicator ,bool addToOut=false)
{
	if(isnull(kroneckerIndicator))
	{
		akron(out_,v1,v2);
		return;
	}
	Eigen::MatrixBase<Derived1>& out = const_cast< Eigen::MatrixBase<Derived1>& >(out_);
	//kroneckerIndicator determines the sizes of the resulting matrix
	out.derived().resize(kroneckerIndicator.rows(),kroneckerIndicator.rows());
	//loop over it
	for (muint_t ir=0;ir<(muint_t)kroneckerIndicator.rows();++ir)
		for (muint_t ic=0;ic<(muint_t)kroneckerIndicator.rows();++ic)
		{
			if (addToOut)
			{
				out(ir,ic) += v1(kroneckerIndicator(ir,0),kroneckerIndicator(ic,0))*v2(kroneckerIndicator(ir,1),kroneckerIndicator(ic,1));
			}
			else
			{
				out(ir,ic) = v1(kroneckerIndicator(ir,0),kroneckerIndicator(ic,0))*v2(kroneckerIndicator(ir,1),kroneckerIndicator(ic,1));
			}
		}//end for
}

/*! Matrix index product diagonal
 * out_{i} = A_{col(i),col(i)} * B_{row(i),row(i)} where row/col indices as provided in a separate list kroneckerIndicator
 * if addToOut:
 * out += A kron B
 *
 * note: if kroneckerIndicator is null, we will revert to a standard kroneckerproduct!
 * \param v1: column covariance
 * \param v2: row covariance
 * \param kroneckerIndicator: [Nrows*Ncos,2], with {index_col,index_row}^N
 */
template <typename Derived1, typename Derived2,typename Derived3,typename Derived4>
inline void aMatrixIndexProduct_diag(const Eigen::MatrixBase<Derived1> & out_, const Eigen::MatrixBase<Derived2>& v1,const Eigen::MatrixBase<Derived3>& v2,const Eigen::MatrixBase<Derived4>& kroneckerIndicator ,bool addToOut=false)
{
	if(isnull(kroneckerIndicator))
	{
		akron_diag(out_,v1,v2);
		return;
	}
	Eigen::MatrixBase<Derived1>& out = const_cast< Eigen::MatrixBase<Derived1>& >(out_);
	//kroneckerIndicator determines the sizes of the resulting matrix
	out.derived().resize(kroneckerIndicator.rows(),1);
	//loop over it
	for (muint_t ir=0;ir<(muint_t)kroneckerIndicator.rows();++ir)
	{
		if (addToOut)
		{
			out(ir,0) += v1(kroneckerIndicator(ir,0),kroneckerIndicator(ir,0))*v2(kroneckerIndicator(ir,1),kroneckerIndicator(ir,1));
		}
		else
		{
			out(ir,0) = v1(kroneckerIndicator(ir,0),kroneckerIndicator(ir,0))*v2(kroneckerIndicator(ir,1),kroneckerIndicator(ir,1));
		}
	}//end for
}


/*
template <typename Derived1, typename Derived2,typename Derived3>
inline Matrixd MatrixIndexProduct_diag(const Eigen::MatrixBase<Derived1>& v1,const Eigen::MatrixBase<Derived2>& v2,const Eigen::MatrixBase<Derived2>& kroneckerIndicator, bool addToOut=false)
{
	MatrixXd out;
	aMatrixIndexProduct_diag(out,v1,v2,kroneckerIndicator,addToOut);
	return out;
}


template <typename Derived1, typename Derived2,typename Derived3>
inline Matrixd MatrixIndexProduct(const Eigen::MatrixBase<Derived1>& v1,const Eigen::MatrixBase<Derived2>& v2,const Eigen::MatrixBase<Derived3>& kroneckerIndicator, bool addToOut=false)
{
	MatrixXd out;
	aMatrixIndexProduct(out,v1,v2,kroneckerIndicator,addToOut);
	return out;
}
*/

template <typename Derived1, typename Derived2,typename Derived3,typename Derived4>
inline MatrixXd MatrixIndexProduct(const Eigen::MatrixBase<Derived2>& v1,const Eigen::MatrixBase<Derived3>& v2,const Eigen::MatrixBase<Derived4>& kroneckerIndicator)
{
	MatrixXd out;
	aMatrixIndexProduct(out,v1,2,kroneckerIndicator);
	return out;
}

template <typename Derived1, typename Derived2,typename Derived3,typename Derived4>
inline MatrixXd MatrixIndexProduct_diag(const Eigen::MatrixBase<Derived2>& v1,const Eigen::MatrixBase<Derived3>& v2,const Eigen::MatrixBase<Derived4>& kroneckerIndicator)
{
	MatrixXd out;
	aMatrixIndexProduct_diag(out,v1,2,kroneckerIndicator);
	return out;
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
inline mfloat_t getVarianceK(const Eigen::MatrixBase<Derived1> & K_) 
{
	//cast out arguments
	//Eigen::MatrixBase<Derived1>& K = const_cast< Eigen::MatrixBase<Derived1>& >(K);
	//ensure that it is a square matrix:
	if (K_.rows()!=K_.cols())
		throw CLimixException("Kernel scaling requires square kernel matrix");

	//diagonal
	mfloat_t c = K_.trace();
	c -= 1.0/K_.rows() * K_.sum();

	mfloat_t scalar = c/(K_.rows()-1);
	return scalar;
}


template <typename Derived1>
inline void VarianceScaleK(const Eigen::MatrixBase<Derived1> & K_) 
{
	//cast out arguments
	Eigen::MatrixBase<Derived1>& K = const_cast< Eigen::MatrixBase<Derived1>& >(K);

	mfloat_t scalar = getVarianceK(K);
	K/=scalar;
}



}
#endif /* MATRIX_HELPER_H_ */
