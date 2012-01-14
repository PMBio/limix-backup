// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <gael.guennebaud@inria.fr>
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

#include "sparse.h"

template<typename SparseMatrixType> void sparse_permutations(const SparseMatrixType& ref)
{
  typedef typename SparseMatrixType::Index Index;

  const Index rows = ref.rows();
  const Index cols = ref.cols();
  typedef typename SparseMatrixType::Scalar Scalar;
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;
  typedef Matrix<int,Dynamic,1> VectorI;
  
  double density = (std::max)(8./(rows*cols), 0.01);
  
  SparseMatrixType mat(rows, cols), up(rows,cols), lo(rows,cols), res;
  DenseMatrix mat_d = DenseMatrix::Zero(rows, cols), up_sym_d, lo_sym_d, res_d;
  
  initSparse<Scalar>(density, mat_d, mat, 0);

  up = mat.template triangularView<Upper>();
  lo = mat.template triangularView<Lower>();
  
  up_sym_d = mat_d.template selfadjointView<Upper>();
  lo_sym_d = mat_d.template selfadjointView<Lower>();
  
  VERIFY_IS_APPROX(mat, mat_d);
  VERIFY_IS_APPROX(up, DenseMatrix(mat_d.template triangularView<Upper>()));
  VERIFY_IS_APPROX(lo, DenseMatrix(mat_d.template triangularView<Lower>()));
  
  PermutationMatrix<Dynamic> p, p_null;
  VectorI pi;
  randomPermutationVector(pi, cols);
  p.indices() = pi;

  
  res = mat.template selfadjointView<Upper>().twistedBy(p_null);
  res_d = up_sym_d;
  VERIFY(res.isApprox(res_d) && "full selfadjoint upper to full");
  
  res = mat.template selfadjointView<Lower>().twistedBy(p_null);
  res_d = lo_sym_d;
  VERIFY(res.isApprox(res_d) && "full selfadjoint lower to full");
  
  
  res = up.template selfadjointView<Upper>().twistedBy(p_null);
  res_d = up_sym_d;
  VERIFY(res.isApprox(res_d) && "upper selfadjoint to full");
  
  res = lo.template selfadjointView<Lower>().twistedBy(p_null);
  res_d = lo_sym_d;
  VERIFY(res.isApprox(res_d) && "lower selfadjoint full");
  
  
  res.template selfadjointView<Upper>() = mat.template selfadjointView<Upper>().twistedBy(p);
  res_d = ((p * up_sym_d) * p.inverse()).eval().template triangularView<Upper>();
  VERIFY(res.isApprox(res_d) && "full selfadjoint upper twisted to upper");
  
  res.template selfadjointView<Upper>() = mat.template selfadjointView<Lower>().twistedBy(p);
  res_d = ((p * lo_sym_d) * p.inverse()).eval().template triangularView<Upper>();
  VERIFY(res.isApprox(res_d) && "full selfadjoint lower twisted to upper");
  
  res.template selfadjointView<Lower>() = mat.template selfadjointView<Lower>().twistedBy(p);
  res_d = ((p * lo_sym_d) * p.inverse()).eval().template triangularView<Lower>();
  VERIFY(res.isApprox(res_d) && "full selfadjoint lower twisted to lower");
  
  res.template selfadjointView<Lower>() = mat.template selfadjointView<Upper>().twistedBy(p);
  res_d = ((p * up_sym_d) * p.inverse()).eval().template triangularView<Lower>();
  VERIFY(res.isApprox(res_d) && "full selfadjoint upper twisted to lower");
  
  
  res.template selfadjointView<Upper>() = up.template selfadjointView<Upper>().twistedBy(p);
  res_d = ((p * up_sym_d) * p.inverse()).eval().template triangularView<Upper>();
  VERIFY(res.isApprox(res_d) && "upper selfadjoint twisted to upper");
  
  res.template selfadjointView<Upper>() = lo.template selfadjointView<Lower>().twistedBy(p);
  res_d = ((p * lo_sym_d) * p.inverse()).eval().template triangularView<Upper>();
  VERIFY(res.isApprox(res_d) && "lower selfadjoint twisted to upper");
  
  res.template selfadjointView<Lower>() = lo.template selfadjointView<Lower>().twistedBy(p);
  res_d = ((p * lo_sym_d) * p.inverse()).eval().template triangularView<Lower>();
  VERIFY(res.isApprox(res_d) && "lower selfadjoint twisted to lower");
  
  res.template selfadjointView<Lower>() = up.template selfadjointView<Upper>().twistedBy(p);
  res_d = ((p * up_sym_d) * p.inverse()).eval().template triangularView<Lower>();
  VERIFY(res.isApprox(res_d) && "upper selfadjoint twisted to lower");

  
  res = mat.template selfadjointView<Upper>().twistedBy(p);
  res_d = (p * up_sym_d) * p.inverse();
  VERIFY(res.isApprox(res_d) && "full selfadjoint upper twisted to full");
  
  res = mat.template selfadjointView<Lower>().twistedBy(p);
  res_d = (p * lo_sym_d) * p.inverse();
  VERIFY(res.isApprox(res_d) && "full selfadjoint lower twisted to full");
  
  res = up.template selfadjointView<Upper>().twistedBy(p);
  res_d = (p * up_sym_d) * p.inverse();
  VERIFY(res.isApprox(res_d) && "upper selfadjoint twisted to full");
  
  res = lo.template selfadjointView<Lower>().twistedBy(p);
  res_d = (p * lo_sym_d) * p.inverse();
  VERIFY(res.isApprox(res_d) && "lower selfadjoint twisted to full");
}

void test_sparse_permutations()
{
  for(int i = 0; i < g_repeat; i++) {
    int s = Eigen::internal::random<int>(1,50);
    CALL_SUBTEST_1(( sparse_permutations(SparseMatrix<double>(8, 8)) ));
    CALL_SUBTEST_2(( sparse_permutations(SparseMatrix<std::complex<double> >(s, s)) ));
    CALL_SUBTEST_1(( sparse_permutations(SparseMatrix<double>(s, s)) ));
  }
}
