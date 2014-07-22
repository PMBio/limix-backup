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

#include "limix/utils/beta.h"
#include "limix/utils/gamma.h"
//#include <math.h>
//does tis work on windows?
#include <cmath>

namespace stats {

	// Beta distribution
	// See: http://en.wikipedia.org/wiki/Beta_distribution
	/// <summary>Beta distribution</summary>
	/// <summary>Cumulative distribution function</summary>
	/// <param name="x">Value at which to compute the cdf</param>
	/// <param name="a">Shape parameter (alpha)</param>
	/// <param name="b">Shape parameter (beta)</param>
	double Beta::Cdf(double x, double a, double b){
		if (a <= 0 || b <= 0){
			printf("Beta.Cdf parameters, a and b, must be > 0");
			throw(1);
		}
		if (x < 0 || x > 1){
			printf("Beta.Cdf parameter x must be between 0 and 1");
			throw(1);
		}
		return Beta::BetaInc(x, a, b);
	}

	/// <summary>Probability distribution function</summary>
	/// <param name="x">Value at which to compute the pdf</param>
	/// <param name="a">Shape parameter (alpha)</param>
	/// <param name="b">Shape parameter (beta)</param>
	double Beta::Pdf(double x, double a, double b){
		if (a <= 0 || b <= 0){
			printf("Beta.Pdf parameters, a and b, must be > 0");
			throw(1);
		}

		if (x > 1) return 0;
		if (x < 0) return 0;

		double lnb = MathFunctions::LogBeta(a, b);
		return exp((a - 1) * log(x) + (b - 1) * log(1 - x) - lnb);
	}



	/// <summary>Mean and variance</summary>
	/// <param name="a">Shape parameter (alpha)</param>
	/// <param name="b">Shape parameter (beta)</param>
	/// <param name="var">Output: variance</param>
	/// <returns>Mean</returns>
	double Beta::Stats(double a, double b, double *var)
	{
		if (a <= 0 || b <= 0){
			printf("Beta.Stats parameters, a and b, must be > 0");
			throw(1);
		}
		double mean = a / (a + b);
		double ab = a + b;
		*var = (a * b) / (ab * ab * (ab + 1));
		return mean;
	}

	/// <summary>Inverse beta cumulative probability distribution</summary>
	/// <param name="p">Probability at which to compute the inverse cdf</param>
	/// <param name="a">Shape parameter (alpha)</param>
	/// <param name="b">Shape parameter (beta)</param>
	double Beta::Inv(double p, double a, double b)
	{
		// We use an iterative search to find x
		// I'm sure there are better ways of doing this

		if (p < 0 || p > 1){
			printf("Beta.Inv parameter p must be between 0 and 1.");
			throw(1);
		}
		if (a <= 0 || b <= 0){
			printf("Beta.Inv parameters a and b must be greater than 0.");
			throw(1);
		}
		if (p == 0) return 0;
		if (p == 1) return 1;

		// We know that x is between 0 and 1, guess in log-odds space
		double guesslo = -709.0;
		double guesshi = 36.0;

		double guess = 0.0;
		double x = 0.0;
		for (int i = 0; i < MAXITER; ++i)
		{
			// Make a guess half way between the lo and hi
			guess = (guesslo + guesshi) / 2;
			x = 1.0 / (1.0 + exp(-guess));
			if (BetaInc(x, a, b) > p)
				guesshi = guess;    // we guessed too high
			else
				guesslo = guess;    // we guessed too low

			// Convergence!
			if (guesshi - guesslo < 1e-10)
				break;
		}

		return x;
	}

	double Beta::BetaInc(double x, double a, double b)
	{
		if (x == 0 || x == 1)
		{
			return x;
		}
		else
		{
			double c = exp(MathFunctions::logGamma(a + b) - MathFunctions::logGamma(a) - MathFunctions::logGamma(b) +
				a * log(x) + b * log(1.0 - x));
			double p;

			if (x < (a + 1) / (a + b + 2.0))
			{
				double cf = BetaCF(x, a, b);
				p = c * cf / a;
			}
			else
			{
				// Use symmetry relation
				double cf = BetaCF(1.0 - x, b, a);
				p = 1 - c* cf / b;
			}

			if (p >= 1.0) p = 1.0;
			return p;
		}
	}


	// Using the modified Lentz's method, described in section 5.2 of 
	// Numerical Recipes
	double Beta::BetaCF(double x, double a, double b)
	{
		double ap1 = a + 1.0;
		double am1 = a - 1.0;
		double m2 = 2.0;
		double ab = a + b;

		double d = 1.0 - ab * x / ap1;
		if (std::fabs(d) < TINY) d = TINY;
		double c = 1;
		d = 1.0 / d;
		double cf = d;

		for (int m = 1; m <= MAXITER; ++m)
		{
			double aj = m * (b - m) * x / ((am1 + m2) * (a + m2));
			d = 1.0 + aj * d;
			if (std::fabs(d) < TINY) d = TINY;
			c = 1.0 + aj / c;
			if (std::fabs(c) < TINY) c = TINY;
			d = 1.0 / d;
			double delta = c * d;
			cf *= delta;

			aj = -(a + m) * (ab + m) * x / ((ap1 + m2) * (a + m2));
			d = 1.0 + aj * d;
			if (std::fabs(d) < TINY) d = TINY;
			c = 1.0 + aj / c;
			if (std::fabs(c) < TINY) c = TINY;
			d = 1.0 / d;
			delta = c * d;
			cf *= delta;

			if (std::fabs(delta - 1) < EPS)
				break;

			m2 += 2;
		}

		return cf;
	}

	double Beta::Eval(double y, double lo, double hi, double s)
	{
		double a = lo;
		double b = hi;
		double fa = y - Gamma::Psi(a) + Gamma::Psi(a + s);
		if (fa == 0) return a;
		double fb = y - Gamma::Psi(b) + Gamma::Psi(b + s);
		if (fb == 0) return b;

		double c = a;
		double fc = fa;

		double tol = 1e-10;

		for (int i = 0; i < MAXITER; ++i)
		{
			double diff = b - a;
			if (std::fabs(fc) < std::fabs(fb))
			{
				a = b;
				b = c;
				c = a;
				fa = fb;
				fb = fc;
				fc = fa;
			}

			double tolact = 2.0 * EPS * std::fabs(b) + tol / 2.0;
			double nextstep = (c - b) / 2;

			if (std::fabs(nextstep) <= tolact || fb == 0.0)
			{
				return b;
			}

			if (std::fabs(diff) >= tolact && std::fabs(fa) > std::fabs(fb))
			{
				double p, q;
				double cb = c - b;
				if (a == c)
				{
					double t1 = fa / fb;
					p = cb*t1;
					q = 1.0 - t1;
				}
				else
				{
					q = fa / fc;
					double t1 = fb / fc;
					double t2 = fb / fa;
					p = t2*(cb*q*(q - t1) - (b - a)*(t1 - 1.0));
					q = (q - 1.0)*(t1 - 1.0)*(t2 - 1.0);
				}

				if (p > 0)
					q = -q;
				else
					p = -p;

				if (p < (0.75*cb*q - std::fabs(tolact*q) / 2) && p < std::fabs(diff*q / 2))
					nextstep = p / q;

			}

			if (std::fabs(nextstep) < tolact)
			{
				if (nextstep > 0)
					nextstep = tolact;
				else
					nextstep = -tolact;
			}

			// Previous approximation
			a = b;
			fa = fb;
			b += nextstep;
			fb = y - Gamma::Psi(b) + Gamma::Psi(b + s);
			if ((fb > 0 && fc > 0) || (fb < 0 && fc < 0))
			{
				c = a;
				fc = fa;
			}

		}

		return b;
	}

	/*
	// Test case: [.9,.1,.3,.23]
	/// <summary>Maximum likely estimate</summary>
	/// <param name="dataIn">Data to fit</param>
	/// <param name="bhat">Shape estimate (beta)</param>
	/// <returns>Shape estimate (alpha)</returns>
	double Beta::MLE(double *dataIn, double *bhat, int data_size)
	{
	double *data = dataIn;
	if (data_size < 2){
	printf("Beta.MLE parameter, data, must be of length > 1");
	throw(1);
	}
	double meanlnp = 0;
	double meanlp1 = 0;

	for (int i = 0; i < data_size; ++i)
	{
	if (data[i] <= 0 || data[i] >= 1){
	printf("Data for Beta.MLE must be > 0 and < 1.");
	throw(1);
	}
	meanlnp += log(data[i]);
	meanlp1 += log(1 - data[i]);
	}

	double cnt = (double)data_size;
	meanlnp /= cnt;
	meanlp1 /= cnt;

	double p = data.Mean();
	double p1 = 1 - p;
	double var = data.Var();
	double pp = p * p1 / var - 1;

	// Starting guesses
	double ahat = pp * p;
	*bhat = pp * p1;

	double toler = 1.0e-6;

	double alo = 0.001;
	double ahi = ahat * 2;
	double blo = 0.001;
	double bhi = *bhat * 2;

	for (int i = 0; i < MAXITER; ++i)
	{
	double ahatnew = Eval(meanlnp, alo, ahi, *bhat);
	double bhatnew = Eval(meanlp1, blo, bhi, ahat);

	double adiff = ahatnew - ahat;
	double bdiff = bhatnew - bhat;
	double dist = sqrt(adiff * adiff + bdiff * bdiff);
	ahat = ahatnew;
	*bhat = bhatnew;
	if (dist <= toler)
	{
	bool fTryAgain = false;
	// If we bump against the value of ahi or bhi, we increase it and
	// try again. It's possible for the change in a or bhat to get very small but
	// not have converged.
	if (std::fabs(ahat - ahi) <= toler)
	{
	ahi *= 2;
	fTryAgain = true;
	}
	if (std::fabs(*bhat - bhi) <= toler)
	{
	bhi *= 2;
	fTryAgain = true;
	}

	if (!fTryAgain)
	break;
	}
	}

	return ahat;
	}
	*/
}//end :stats