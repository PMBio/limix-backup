#if !defined(FisherF_h)
#define FisherF_h
//#include "../lmm.h"
class FisherF{
   public:
      static double Cdf(double x, double v1, double v2);
      static double Pdf(double x, double v1, double v2);
      static double Inv(double p, double v1, double v2);
      static double Stats(double v1, double v2, double *var);
};

#endif //Fisher_F_h
