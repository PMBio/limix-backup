#if !defined(Gamma_h)
#define Gamma_h
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

namespace stats {
	class Gamma
	{
	public:
		static double gammaQ(double x, double a);
		static double gammaIncLower(double x, double a);
		static double gammaIncUpper(double x, double a);
		static double gammaP(double x, double a);
		static double cdf(double x, double k, double theta);
		static double inv(double p, double k, double theta);
		static double Psi(double k);
	};
}//end :stats
#endif  //Gamma_h
