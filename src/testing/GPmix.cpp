//============================================================================
// Name        : GPmix.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "gpmix/gp/gp_base.h"
#include "gpmix/types.h"
#include "gpmix/matrix/matrix_helper.h"
#include "gpmix/likelihood/likelihood.h"
#include "gpmix/covar/linear.h"
//#include "gpmix/covar/se.h"
#include "gpmix/gp/gp_base.h"

using namespace std;

#ifndef PI
#define PI 3.14159265358979323846
#endif

int main() {
	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!

	uint_t nX=100;

	float_t xmin=1.0;
	float_t xmax = 2.5*PI;

	uint_t ntrain =15;

	gpmix::MatrixXd x = gpmix::VectorXd::LinSpaced(ntrain,xmin,xmax);
	uint_t dimX = x.cols();
	gpmix::MatrixXd X = gpmix::VectorXd::LinSpaced(nX,0,10.0);

	//float_t C = 2.0;
	float_t sigma = 0.01;
	//float_t b = 0.0;
	gpmix::MatrixXd y(ntrain,1);

	for (uint_t i=0;i<ntrain;++i)
	{
		y(i) = sin((float_t)x(i));//WARNING: cast as float_t
	}


	//y+=x.sin();
	uint_t dimY = y.cols();
	y += sigma * gpmix::randn(ntrain,dimY);
	float_t meanY = y.mean();

	y.array()-=meanY;


	//likelihood model
	gpmix::CLikNormalIso lik = gpmix::CLikNormalIso();

	//parameters of likelihood model
	gpmix::LikParams likparams(1);
	likparams << 1.0;

	//gpmix::ACovarianceFunction cov;
#if 1
	//linear kernel + params
	gpmix::CCovLinearISO cov = gpmix::CCovLinearISO();
	gpmix::CovarParams covparams(1);
	covparams << 1.0;
#else
	//sqex noch nicht umgestellt
	//squared exponential kernel + params
	gpmix::CCovSqexpARD cov(1);
	gpmix::CovarParams covparams(2,1);
	covparams << 1.0 , 1.0;
#endif
	//TODO: build parameters object
	//gpmix::CGPHyperParams gpparams;
	//gpparams.set("lik", likparams);//set((string)"lik",(MatrixXd)likparams);
	//gpparams.set("covar", covparams);

	//MatrixXd likrecover = gpparams.get("lik");

	//create GP_base
	//gpmix::ACovarianceFunction& mcov  = cov;

	gpmix::CGPbase gp(cov,lik);

	//set data of GP
	gp.set_data(y);
	//gp.set_params(gpparams);

	//evaluate negative log-likelihood
	float_t nLL = gp.LML();

	//evaluate gradient
	//gpmix::CGPHyperParams grad = gp.LMLgrad();


	//TODO: optimize parameters



	cout << "x:" << endl;
	cout << x << endl;
	cout << "done."<<endl;
	cout << "y:" << endl;
	cout << y << endl;
	cout << "done."<<dimX<<endl;
	cout << "nLL: "<<nLL <<endl;
	/*gpmix::VectorXs names = grad.getNames();
	for (uint_t i = 0; i< (uint_t)names.rows(); ++i)
	{
		string curname =names(i);
		gpmix::MatrixXd curgrad = grad.get(curname);
		cout<< curname <<":" << curgrad << "\n";
	}
	//TODO: predict
	gpmix::MatrixXd meanStar = gp.predictMean(X);
	cout << "mean:\n" << meanStar <<endl;

	gpmix::MatrixXd varStar = gp.predictVar(X);
	cout << "var:\n" << varStar <<endl;

*/

	return 0;
}
