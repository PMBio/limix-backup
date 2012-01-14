// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <g.gael@free.fr>
//
// Eigen is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// Alternatively, you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of
// the License, or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License or the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License and a copy of the GNU General Public License along with
// Eigen. If not, see <http://www.gnu.org/licenses/>.

#define EIGEN_NO_DEPRECATED_WARNING

#include "sparse.h"
#include <Eigen/SparseExtra>

#ifdef EIGEN_CHOLMOD_SUPPORT
#include <Eigen/CholmodSupport>
#endif

template<typename Scalar,typename Index> void sparse_ldlt(int rows, int cols)
{
  static bool odd = true;
  odd = !odd;
  double density = (std::max)(8./(rows*cols), 0.01);
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;
  typedef Matrix<Scalar,Dynamic,1> DenseVector;
  typedef SparseMatrix<Scalar,ColMajor,Index> SparseMatrixType;
  
  SparseMatrixType m2(rows, cols);
  DenseMatrix refMat2(rows, cols);

  DenseVector b = DenseVector::Random(cols);
  DenseVector refX(cols), x(cols);

  initSparse<Scalar>(density, refMat2, m2, ForceNonZeroDiag|MakeUpperTriangular, 0, 0);
  
  SparseMatrixType m3 = m2 * m2.adjoint(), m3_lo(rows,rows), m3_up(rows,rows);
  DenseMatrix refMat3 = refMat2 * refMat2.adjoint();
  
  refX = refMat3.template selfadjointView<Upper>().ldlt().solve(b);
  typedef SparseMatrix<Scalar,Upper|SelfAdjoint,Index> SparseSelfAdjointMatrix;
  x = b;
  SparseLDLT<SparseSelfAdjointMatrix> ldlt(m3);
  if (ldlt.succeeded())
    ldlt.solveInPlace(x);
  else
    std::cerr << "warning LDLT failed\n";

  VERIFY_IS_APPROX(refMat3.template selfadjointView<Upper>() * x, b);
  VERIFY(refX.isApprox(x,test_precision<Scalar>()) && "LDLT: default");
  
#ifdef EIGEN_CHOLMOD_SUPPORT
  {
    x = b;
    SparseLDLT<SparseSelfAdjointMatrix, Cholmod> ldlt2(m3);
    if (ldlt2.succeeded())
    {
      ldlt2.solveInPlace(x);
      VERIFY_IS_APPROX(refMat3.template selfadjointView<Upper>() * x, b);
      VERIFY(refX.isApprox(x,test_precision<Scalar>()) && "LDLT: cholmod solveInPlace");
      
      x = ldlt2.solve(b);
      VERIFY_IS_APPROX(refMat3.template selfadjointView<Upper>() * x, b);
      VERIFY(refX.isApprox(x,test_precision<Scalar>()) && "LDLT: cholmod solve");
    }
    else
      std::cerr << "warning LDLT failed\n";
  }
#endif
  
  // new Simplicial LLT
  
  
  // new API
  {
    SparseMatrixType m2(rows, cols);
    DenseMatrix refMat2(rows, cols);

    DenseVector b = DenseVector::Random(cols);
    DenseVector ref_x(cols), x(cols);
    DenseMatrix B = DenseMatrix::Random(rows,cols);
    DenseMatrix ref_X(rows,cols), X(rows,cols);

    initSparse<Scalar>(density, refMat2, m2, ForceNonZeroDiag|MakeLowerTriangular, 0, 0);

    for(int i=0; i<rows; ++i)
      m2.coeffRef(i,i) = refMat2(i,i) = internal::abs(internal::real(refMat2(i,i)));
    
    
    SparseMatrixType m3 = m2 * m2.adjoint(), m3_lo(rows,rows), m3_up(rows,rows);
    DenseMatrix refMat3 = refMat2 * refMat2.adjoint();
    
    m3_lo.template selfadjointView<Lower>().rankUpdate(m2,0);
    m3_up.template selfadjointView<Upper>().rankUpdate(m2,0);
    
    // with a single vector as the rhs
    ref_x = refMat3.template selfadjointView<Lower>().llt().solve(b);

    x = SimplicialCholesky<SparseMatrixType, Lower>().setMode(odd ? SimplicialCholeskyLLt : SimplicialCholeskyLDLt).compute(m3).solve(b);
    VERIFY(ref_x.isApprox(x,test_precision<Scalar>()) && "SimplicialCholesky: solve, full storage, lower, single dense rhs");
    
    x = SimplicialCholesky<SparseMatrixType, Upper>().setMode(odd ? SimplicialCholeskyLLt : SimplicialCholeskyLDLt).compute(m3).solve(b);
    VERIFY(ref_x.isApprox(x,test_precision<Scalar>()) && "SimplicialCholesky: solve, full storage, upper, single dense rhs");
    
    x = SimplicialCholesky<SparseMatrixType, Lower>(m3_lo).solve(b);
    VERIFY(ref_x.isApprox(x,test_precision<Scalar>()) && "SimplicialCholesky: solve, lower only, single dense rhs");
    
    x = SimplicialCholesky<SparseMatrixType, Upper>(m3_up).solve(b);
    VERIFY(ref_x.isApprox(x,test_precision<Scalar>()) && "SimplicialCholesky: solve, upper only, single dense rhs");
    
    
    // with multiple rhs
    ref_X = refMat3.template selfadjointView<Lower>().llt().solve(B);

    X = SimplicialCholesky<SparseMatrixType, Lower>().setMode(odd ? SimplicialCholeskyLLt : SimplicialCholeskyLDLt).compute(m3).solve(B);
    VERIFY(ref_X.isApprox(X,test_precision<Scalar>()) && "SimplicialCholesky: solve, full storage, lower, multiple dense rhs");
    
    X = SimplicialCholesky<SparseMatrixType, Upper>().setMode(odd ? SimplicialCholeskyLLt : SimplicialCholeskyLDLt).compute(m3).solve(B);
    VERIFY(ref_X.isApprox(X,test_precision<Scalar>()) && "SimplicialCholesky: solve, full storage, upper, multiple dense rhs");
    
    
    // with a sparse rhs
    SparseMatrixType spB(rows,cols), spX(rows,cols);
    B.diagonal().array() += 1;
    spB = B.sparseView(0.5,1);
    
    ref_X = refMat3.template selfadjointView<Lower>().llt().solve(DenseMatrix(spB));

    spX = SimplicialCholesky<SparseMatrixType, Lower>(m3).solve(spB);
    VERIFY(ref_X.isApprox(spX.toDense(),test_precision<Scalar>()) && "LLT: SimplicialCholesky solve, multiple sparse rhs");
//     
    spX = SimplicialCholesky<SparseMatrixType, Upper>(m3).solve(spB);
    VERIFY(ref_X.isApprox(spX.toDense(),test_precision<Scalar>()) && "LLT: SimplicialCholesky solve, multiple sparse rhs");
  }
  
  
  
//   for(int i=0; i<rows; ++i)
//     m2.coeffRef(i,i) = refMat2(i,i) = internal::abs(internal::real(refMat2(i,i)));
// 
//   refX = refMat2.template selfadjointView<Upper>().ldlt().solve(b);
//   typedef SparseMatrix<Scalar,Upper|SelfAdjoint> SparseSelfAdjointMatrix;
//   x = b;
//   SparseLDLT<SparseSelfAdjointMatrix> ldlt(m2);
//   if (ldlt.succeeded())
//     ldlt.solveInPlace(x);
//   else
//     std::cerr << "warning LDLT failed\n";
// 
//   VERIFY_IS_APPROX(refMat2.template selfadjointView<Upper>() * x, b);
//   VERIFY(refX.isApprox(x,test_precision<Scalar>()) && "LDLT: default");


}

void test_sparse_ldlt()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( (sparse_ldlt<double,int>(8, 8)) );
    CALL_SUBTEST_1( (sparse_ldlt<double,long int>(8, 8)) );
    int s = internal::random<int>(1,300);
    CALL_SUBTEST_2( (sparse_ldlt<std::complex<double>,int>(s,s)) );
    CALL_SUBTEST_1( (sparse_ldlt<double,int>(s,s)) );
  }
}
