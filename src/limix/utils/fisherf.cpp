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

#include "limix/utils/fisherf.h"
#include "limix/utils/beta.h"

#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <math.h>

namespace stats {
	// F distribution
	// See: http://en.wikipedia.org/wiki/F-distribution
	/// <summary>Cumulative distribution function</summary>
	/// <param name="x">Value at which to compute the cdf</param>
	/// <param name="v1">Degree of freedom parameter</param>
	/// <param name="v2">Degree of freedom parameter</param>
	double FisherF::Cdf(double x, double v1, double v2){
		if (x == 0) return 0;

		if (x < 0){
			printf("Fisher_F::Cdf parameter x must be > 0.");
			throw(1);
		}
		if (v1 <= 0 || v2 <= 0){
			printf("Fisher_F::Cdf parameters v1 and v2 must be > 0.");
			throw(1);
		}


		double vx = v1 * x;
		double k = vx / (vx + v2);
		return Beta::BetaInc(k, v1 / 2, v2 / 2);
	}

	/// <summary>Probability distribution function</summary>
	/// <param name="x">Value at which to compute the pdf</param>
	/// <param name="v1">Degree of freedom parameter</param>
	/// <param name="v2">Degree of freedom parameter</param>
	double FisherF::Pdf(double x, double v1, double v2){
		if (x == 0) return 0;

		if (x < 0){
			printf("Fisher_F::Pdf parameter x must be > 0.");
			throw(1);
		}
		if (v1 <= 0 || v2 <= 0){
			printf("Fisher_F::Pdf parameters v1 and v2 must be > 0.");
			throw(1);
		}

		double d1 = v1 * x;
		double d = d1 / (d1 + v2);

		double v1half = v1 / 2.0;
		double v2half = v2 / 2.0;

		return exp(v1half * log(d) + v2half * log(1 - d) - log(x) - MathFunctions::LogBeta(v1half, v2half));
	}

	/// <summary>Inverse cumulative distribution function</summary>
	/// <param name="p">Probability at which to compute the inverse cdf</param>
	/// <param name="v1">Degree of freedom parameter</param>
	/// <param name="v2">Degree of freedom parameter</param>
	double FisherF::Inv(double p, double v1, double v2){
		if (p < 0 || p > 1){
			printf("Fisher_F::Inv parameter p must be between 0 and 1.");
			throw(1);
		}

		if (v1 <= 0 || v2 <= 0){
			printf("Fisher_F::Inv parameters v1 and v2 must be > 0.");
			throw(1);
		}

		if (p == 0) return 0;
		if (p == 1) return std::numeric_limits<double>::infinity();

		double x = Beta::Inv(1 - p, v2 / 2, v1 / 2);
		return (v2 - v2 * x) / (v1 * x);
	}

	/// <summary>Mean and variance</summary>
	/// <param name="v1">Degree of freedom parameter</param>
	/// <param name="v2">Degree of freedom parameter</param>
	/// <param name="var">Output: variance</param>
	/// <returns>Mean</returns>
	double FisherF::Stats(double v1, double v2, double *var){
		if (v2 <= 2){
			printf("Fisher_F::Stats parameter v2 must be > 2.");
			throw(1);
		}

		double mu = v2 / (v2 - 2);

		*var = std::numeric_limits<double>::quiet_NaN();
		if (v2 > 4)
			*var = 2 * v2 * v2 * (v1 + v2 - 2) / (v1 * (v2 - 2) * (v2 - 2) * (v2 - 4));
		return mu;
	}
}//end :stats