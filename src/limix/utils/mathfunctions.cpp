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


#include "limix/utils/mathfunctions.h"

#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <math.h>

const double _PI = 2.0*acos(0.0);          // now defined in FastLmmC.h
const double _halflog2pi=(double)0.5*log((double)2.0*_PI);
const double coeffsForLogGamma[] = { 12.0, -360.0, 1260.0, -1680.0, 1188.0 };

const double MathFunctions::eps_rank=(double)3E-8;

// Gamma and LogGamma

// Use GalenA's implementation of LogGamma - it's faster!
/// <summary>Returns the log of the gamma function</summary>
/// <param name="x">Argument of function</param>
/// <returns>Log Gamma(x)</returns>
/// <remarks>Accurate to eight digits for all x.</remarks>
double MathFunctions::logGamma(double x)
{
   if (x <= (double)0.0){
      printf("LogGamma arg=%f must be > 0.",x);
      throw(1);
   }

   double res = (double)0.0;
   if (x < (double)6.0)
   {
         int toAdd = (int)floor(7 - x);
         double v2 = (double)1.0;
         for (int i = 0; i < toAdd; i++)
         {
            v2 *= (x + i);
         }
         res = -log(v2);
         x += toAdd;
   }
   x -= (double)1.0;

   res += _halflog2pi + (x + (double)0.5) * log(x) - x;

   // correction terms
   double xSquared = x * x;
   double pow = x;
   for (int i=0; i<5; ++i)   //the length of the coefficient array is 5.
   {
         double newRes = res + (double)1.0 / (coeffsForLogGamma[i] * pow);
         if (newRes == res)
         {
            return res;
         }
         res = newRes;
         pow *= xSquared;
   }

   return res;
}

// Beta and LogBeta
/// <summary>Computes the log beta function</summary>
double MathFunctions::LogBeta(double x, double y)
{
   if (x <= 0.0 || y <= 0.0){
      printf("LogBeta args must be > 0.");
      throw(1);
   }
   return MathFunctions::logGamma(x) + MathFunctions::logGamma(y) - MathFunctions::logGamma(x + y);
}

int MathFunctions::randperm(const size_t n, const double *in, double *out){
   if (n>0){
      out[0] = in[0];
      size_t j;
      for (size_t i=1; i<n; ++i){
         j = rand() % (i+1);
         out[i] = out[j];
         out[j] = in[i];
      }
   }
   return 0;
}

int MathFunctions::randperm(const size_t n, size_t *randperm){
   if (n>0){
      randperm[0] = 0;
      size_t j;
      for (size_t i=1; i<n; ++i){
         j = rand() % (i+1);
         randperm[i] = randperm[j];
         randperm[j] = i;
      }
   }
   return 0;
}
