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

#if !defined(FisherF_h)
#define FisherF_h
//#include "../lmm.h"
namespace stats {
	class FisherF{
	public:
		static double Cdf(double x, double v1, double v2);
		static double Pdf(double x, double v1, double v2);
		static double Inv(double p, double v1, double v2);
		static double Stats(double v1, double v2, double *var);
	};
}//end:stats
#endif //Fisher_F_h
