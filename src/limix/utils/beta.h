#if !defined(Beta_h)
#define Beta_h

#include "limix/utils/mathfunctions.h"
namespace stats {
	class Beta{
	public:
		static double BetaCF(double x, double a, double b);
		static double BetaInc(double x, double a, double b);
		static double Cdf(double x, double a, double b);
		static double Pdf(double x, double a, double b);
		static double Stats(double a, double b, double* var);
		static double Inv(double p, double a, double b);
		static double Eval(double y, double lo, double hi, double s);
		//static double MLE(double *dataIn, double *bhat);
	};


	const int MAXITER = 400;
	const double TINY = (double)1.0e-30;
	const double EPS = (double)2.2204e-016;
}//end :stats
#endif //Beta_h
