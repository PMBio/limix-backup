/*
*******************************************************************
*
*    Copyright (c) Microsoft. All rights reserved.
*    This code is licensed under the Apache License, Version 2.0.
*    THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
*    ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
*    IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
*    PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
*
******************************************************************
*/
#if !defined(MathFunctions_h)
#define MathFunctions_h

#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <limits>

class MathFunctions
{

public:
   static double logGamma(double x);
   static double LogBeta(double x, double y);
   static int leastSquares2by2SymLower(double *X, const double *y, double *result, double *f_stat, double *logDetXX);
   static double det3by3symmL(const double *X, double *f_stat);
   static double det3by3symmY0L(const double *X, const double *y, double *f_stat);
   static double det3by3symmY1L(const double *X, const double *y);
   static double det3by3symmY2L(const double *X, const double *y);
   static int solve3by3symmL(const double *X, const double *y, double *beta, double *f_stat);
   static int randperm(const size_t n, const double *in, double *out);
   static int randperm(const size_t n, size_t *randperm);
   static const double eps_rank;
};

//solve the quadratic least squares system of a positive semi-definite symmetric matrix X in lower storage. On exit X holds the inverse of X and res holds the solution.
//returns the rank of X
inline int MathFunctions::leastSquares2by2SymLower(double *X, const double *y, double *result, double *f_stat, double *logDetXX){

   double determinant=X[0]*X[3]-X[1]*X[1];
   if ( determinant > MathFunctions::eps_rank ){//full rank case. just solve
      //inverse of X
      *logDetXX=log(determinant);
      result[0]=(y[0]*X[3]-y[1]*X[1])/determinant;
      result[1]=(y[1]*X[0]-X[1]*y[0])/determinant;
      f_stat[0]=result[0]*result[0]*determinant/X[3];
      f_stat[1]=result[1]*result[1]*determinant/X[1];

      return 2;//rank2
   }else{// see http://www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/index.html for derivations of Eigenvalues and eigenvectors
      *logDetXX=-std::numeric_limits<double>::infinity();
      X[3]+=X[0]; //In this case the trace (stored in X[3] equals the nonzero eigenvalue L_1 in the derivation
      if (X[3] < MathFunctions::eps_rank){//diagonal entries are always non-negative
         result[0] = 0.0;
         result[1] = 0.0;
         f_stat[0] = 0.0;
         f_stat[1] = 0.0;
         return 0;//rank 0
      }
      else{
         if (X[1] < MathFunctions::eps_rank  && X[1] > -MathFunctions::eps_rank){//numerically zero?
            result[0]=y[0]/X[3];
            result[1]=y[1]/X[3];
            f_stat[0]=y[0]*y[0];
            f_stat[1]=y[1]*y[1];
         }else{
            X[3]*= (X[0]*X[0]+X[1]*X[1]);
            X[2] = (X[0]*y[0]+X[1]*y[1])/X[3]; //cache y[0]
            result[0] = X[2]*X[0];
            result[1] = X[2]*X[1];
            f_stat[0] = result[0] * result[0] * X[3] / X[0];
            f_stat[0] = result[0] * result[0] * X[3] / X[1];
         }
         return 1;//rank1
      }
   }
}

//X is a 3-by-3 symmetrix matrix with lower half filled in.
//y is a 3-by-1 vector
//beta is a 3-by-1 vector holding the result
//solve a 3-by-3 system of linear equations
inline int MathFunctions::solve3by3symmL(const double *X, const double *y, double *beta, double *f_stat){
   double det=MathFunctions::det3by3symmL(X, f_stat);
   if (det<MathFunctions::eps_rank){ //low rank
      return 0;
   }else{
      beta[0]=MathFunctions::det3by3symmY0L(X,y, f_stat);
      beta[1]=MathFunctions::det3by3symmY1L(X,y)/det;
      beta[2]=MathFunctions::det3by3symmY2L(X,y)/det;
      f_stat[0]=beta[0]*beta[0]/f_stat[0];
      f_stat[1]=beta[1]*beta[1]/f_stat[1];
      f_stat[2]=beta[2]*beta[2]/f_stat[2];
      return 3;
   }
}

inline double MathFunctions::det3by3symmL(const double *X, double *f_stat){
   f_stat[0]=(X[4]*X[8]-X[5]*X[5]);
   f_stat[1]=(X[1]*X[8]-X[5]*X[2]);
   f_stat[2]=(X[1]*X[5]-X[4]*X[2]);
   double det=(X[0]*f_stat[0]-X[1]*f_stat[1]+X[2]*f_stat[2]);
   f_stat[0]/=det;
   f_stat[1]/=det;
   f_stat[2]/=det;
   return det;
}
inline double MathFunctions::det3by3symmY0L(const double *X,const double *y, double *f_stat){
   return (y[0]*f_stat[0]-y[1]* f_stat[1]+y[2]* f_stat[2]);
}
inline double MathFunctions::det3by3symmY1L(const double *X, const double *y){
   return (X[0]*(y[1]*X[8]-y[2]*X[5])-X[1]*(y[0]*X[8]-y[2]*X[2])+X[2]*(y[0]*X[5]-y[1]*X[2]));
}
inline double MathFunctions::det3by3symmY2L(const double *X, const double *y){
   return (X[0]*(X[4]*y[2]-X[5]*y[1])-X[1]*(X[1]*y[2]-X[5]*y[0])+X[2]*(X[1]*y[1]-X[4]*y[0]));
}

#endif
