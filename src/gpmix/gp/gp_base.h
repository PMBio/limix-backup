/*
 * gp_base.h
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */

#ifndef GP_BASE_H_
#define GP_BASE_H_

#include <gpmix/covar/covariance.h>
#include <gpmix/likelihood/likelihood.h>
#include <string>
#include <map>
#include <gpmix/types.h>
using namespace std;

namespace gpmix {

class CGPHyperParams {

protected:
   
	MatrixXd param_array;
	map<string,MatrixXd> param_map;

public:

	CGPHyperParams()
	{
		//empty constructur
	}
	//from a list of params

	inline VectorXd getParamArray()
	{
		return param_array;
	}

	inline void setParamArray(MatrixXd& param)
	{
		//TODO: check that length is ok
		this->param_array = param;
	}

	void set(string name, MatrixXd value)
	{//why does the following not work? void set(string& name, MatrixXd& value)
		param_map[name]=value;
	}

	inline MatrixXd get(const string& name)
	{
		return param_map[name];
	}

	inline VectorXs getNames()
	{
		VectorXs ret = VectorXs(param_map.size());
		map<string,MatrixXd>::iterator it=param_map.begin();
		for (uint_t i=0; i<param_map.size();++i)
		{
			ret(i)=(*it).first;
			++it;
		}
		return VectorXs(1,1);
	}

	inline void clear()
	{
		this->param_map.clear();
	};
};

class CGPCache {
protected:
	CGPHyperParams cachedparams;
	bool is_cached(CGPHyperParams params);
public:
	CGPCache();
	~CGPCache();
	MatrixXd Kinv(CGPHyperParams params, MatrixXd X, bool check_passed = false, bool is_checked = false);
	MatrixXd KinvY(CGPHyperParams params, MatrixXd X, MatrixXd Y, bool check_passed = false, bool is_checked = false);
	Eigen::LDLT<gpmix::MatrixXd> CholK(CGPHyperParams params, MatrixXd X, bool check_passed = false, bool is_checked = false);
	void clear();
};

class CGPbase {
protected:

	MatrixXd X;    //training inputs
	MatrixXd Y;    //training targets
	VectorXd meanY; //mean of training targets
	Eigen::LDLT<gpmix::MatrixXd> chol;
	MatrixXd KinvY;

	ACovarianceFunction& covar;//Covariance function
	ALikelihood& lik;          //likelihood model

	void getCovariances(CGPHyperParams& hyperparams);

	//virtual float_t _LML_covar(CGPHyperParams& parmas);      //the log-likelihood without the prior
	//virtual VectorXd _LMLgrad_covar(CGPHyperParams& params); //the gradient of the log-likelihood without the prior

public:
	CGPbase(ACovarianceFunction& covar, ALikelihood& lik);
	virtual ~CGPbase();

//TODO: add interface that is suitable for optimizer
// virtual double LML(double* params);
// virtual void LML(double* params, double* gradients);
	virtual void set_data(MatrixXd X, MatrixXd Y);

	virtual float_t LML(CGPHyperParams& hyperparams);        //the log-likelihood (+ log-prior)
	virtual CGPHyperParams LMLgrad(CGPHyperParams& hyperparams);   //the gradient of the log-likelihood (+ log-prior)
   
	inline uint_t get_samplesize(){return this->Y.rows();} //get the number of training data samples
	inline uint_t get_input_dimension(){return this->X.cols();} //get the number of training data samples
	inline uint_t get_target_dimension(){return this->Y.cols();} //get the dimension of the target data
};

} /* namespace gpmix */
#endif /* GP_BASE_H_ */
