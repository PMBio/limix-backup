// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2011 Gael Guennebaud <gael.guennebaud@inria.fr>
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

#ifndef EIGEN_TESTSPARSE_H
#define EIGEN_TESTSPARSE_H

#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

#include "main.h"

#if EIGEN_GNUC_AT_LEAST(4,0) && !defined __ICC && !defined(__clang__)

#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

#include <tr1/unordered_map>
#define EIGEN_UNORDERED_MAP_SUPPORT
namespace std {
  using std::tr1::unordered_map;
}
#endif

#ifdef EIGEN_GOOGLEHASH_SUPPORT
  #include <google/sparse_hash_map>
#endif

#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <Eigen/Sparse>

enum {
  ForceNonZeroDiag = 1,
  MakeLowerTriangular = 2,
  MakeUpperTriangular = 4,
  ForceRealDiag = 8
};

/* Initializes both a sparse and dense matrix with same random values,
 * and a ratio of \a density non zero entries.
 * \param flags is a union of ForceNonZeroDiag, MakeLowerTriangular and MakeUpperTriangular
 *        allowing to control the shape of the matrix.
 * \param zeroCoords and nonzeroCoords allows to get the coordinate lists of the non zero,
 *        and zero coefficients respectively.
 */
template<typename Scalar,int Opt1,int Opt2,typename Index> void
initSparse(double density,
           Matrix<Scalar,Dynamic,Dynamic,Opt1>& refMat,
           SparseMatrix<Scalar,Opt2,Index>& sparseMat,
           int flags = 0,
           std::vector<Vector2i>* zeroCoords = 0,
           std::vector<Vector2i>* nonzeroCoords = 0)
{
  enum { IsRowMajor = SparseMatrix<Scalar,Opt2,Index>::IsRowMajor };
  sparseMat.setZero();
  //sparseMat.reserve(int(refMat.rows()*refMat.cols()*density));
  sparseMat.reserve(VectorXi::Constant(IsRowMajor ? refMat.rows() : refMat.cols(), (1.5*density)*(IsRowMajor?refMat.cols():refMat.rows())));
  
  for(int j=0; j<sparseMat.outerSize(); j++)
  {
    //sparseMat.startVec(j);
    for(int i=0; i<sparseMat.innerSize(); i++)
    {
      int ai(i), aj(j);
      if(IsRowMajor)
        std::swap(ai,aj);
      Scalar v = (internal::random<double>(0,1) < density) ? internal::random<Scalar>() : Scalar(0);
      if ((flags&ForceNonZeroDiag) && (i==j))
      {
        v = internal::random<Scalar>()*Scalar(3.);
        v = v*v + Scalar(5.);
      }
      if ((flags & MakeLowerTriangular) && aj>ai)
        v = Scalar(0);
      else if ((flags & MakeUpperTriangular) && aj<ai)
        v = Scalar(0);

      if ((flags&ForceRealDiag) && (i==j))
        v = internal::real(v);

      if (v!=Scalar(0))
      {
        //sparseMat.insertBackByOuterInner(j,i) = v;
        sparseMat.insertByOuterInner(j,i) = v;
        if (nonzeroCoords)
          nonzeroCoords->push_back(Vector2i(ai,aj));
      }
      else if (zeroCoords)
      {
        zeroCoords->push_back(Vector2i(ai,aj));
      }
      refMat(ai,aj) = v;
    }
  }
  //sparseMat.finalize();
}

template<typename Scalar,int Opt1,int Opt2,typename Index> void
initSparse(double density,
           Matrix<Scalar,Dynamic,Dynamic, Opt1>& refMat,
           DynamicSparseMatrix<Scalar, Opt2, Index>& sparseMat,
           int flags = 0,
           std::vector<Vector2i>* zeroCoords = 0,
           std::vector<Vector2i>* nonzeroCoords = 0)
{
  enum { IsRowMajor = DynamicSparseMatrix<Scalar,Opt2,Index>::IsRowMajor };
  sparseMat.setZero();
  sparseMat.reserve(int(refMat.rows()*refMat.cols()*density));
  for(int j=0; j<sparseMat.outerSize(); j++)
  {
    sparseMat.startVec(j); // not needed for DynamicSparseMatrix
    for(int i=0; i<sparseMat.innerSize(); i++)
    {
      int ai(i), aj(j);
      if(IsRowMajor)
        std::swap(ai,aj);
      Scalar v = (internal::random<double>(0,1) < density) ? internal::random<Scalar>() : Scalar(0);
      if ((flags&ForceNonZeroDiag) && (i==j))
      {
        v = internal::random<Scalar>()*Scalar(3.);
        v = v*v + Scalar(5.);
      }
      if ((flags & MakeLowerTriangular) && aj>ai)
        v = Scalar(0);
      else if ((flags & MakeUpperTriangular) && aj<ai)
        v = Scalar(0);

      if ((flags&ForceRealDiag) && (i==j))
        v = internal::real(v);

      if (v!=Scalar(0))
      {
        sparseMat.insertBackByOuterInner(j,i) = v;
        if (nonzeroCoords)
          nonzeroCoords->push_back(Vector2i(ai,aj));
      }
      else if (zeroCoords)
      {
        zeroCoords->push_back(Vector2i(ai,aj));
      }
      refMat(ai,aj) = v;
    }
  }
  sparseMat.finalize();
}

template<typename Scalar> void
initSparse(double density,
           Matrix<Scalar,Dynamic,1>& refVec,
           SparseVector<Scalar>& sparseVec,
           std::vector<int>* zeroCoords = 0,
           std::vector<int>* nonzeroCoords = 0)
{
  sparseVec.reserve(int(refVec.size()*density));
  sparseVec.setZero();
  for(int i=0; i<refVec.size(); i++)
  {
    Scalar v = (internal::random<double>(0,1) < density) ? internal::random<Scalar>() : Scalar(0);
    if (v!=Scalar(0))
    {
      sparseVec.insertBack(i) = v;
      if (nonzeroCoords)
        nonzeroCoords->push_back(i);
    }
    else if (zeroCoords)
        zeroCoords->push_back(i);
    refVec[i] = v;
  }
}

#endif // EIGEN_TESTSPARSE_H
