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
	
	VectorXd getParamArray()
	{
		if (this->param_array.cols()==0)
		{
			uint_t nparams = this->getNparams();

			this->param_array = VectorXd(nparams);
			uint_t ncurrent=0;
			map<string,MatrixXd>::iterator it=param_map.begin();
			for (uint_t i=0; i<param_map.size();++i)
			{
				ncurrent+=(*it).second.array().rows();
				this->param_array.block(ncurrent,0,(*it).second.array().rows(),1)=(*it).second.array();
				++it;
			}

		}
		return param_array;
	}

	uint_t getNparams()
	{
		uint_t nparams=0;
		map<string,MatrixXd>::iterator it=param_map.begin();
		for (uint_t i=0; i<param_map.size();++i)
		{
			nparams+=(*it).second.array().rows();
			++it;
		}
		return nparams;
	}

	void setParamArray(VectorXd& param)
	{
		uint_t nparams = this->getNparams();

		if ((uint_t)param.rows()!= nparams)//WARNING: uint_t conversion
		{
			ostringstream os;
			os << "Parameter dimensions don't match. param.rows() = "<< param.rows() << ", number parameters = "<< nparams;
			throw gpmix::CGPMixException(os.str());
		}

		this->param_array = param;

		this->param_array = VectorXd(nparams);
		uint_t ncurrent=0;
		map<string,MatrixXd>::iterator it=param_map.begin();
		for (uint_t i=0; i<param_map.size();++i)
		{
			ncurrent+=(*it).second.array().rows();
			(*it).second.array() = this->param_array.block(ncurrent,0,(*it).second.array().rows(),1);
			++it;
		}

	}

	void set(string name, MatrixXd value)
	{//why does the following not work? void set(string& name, MatrixXd& value)
		param_map[name]=value;
	}

	inline MatrixXd get(const string& name)
	{
		return param_map[name];
	}

	VectorXs getNames()
	{
		VectorXs ret = VectorXs(param_map.size());
		map<string,MatrixXd>::iterator it=param_map.begin();
		for (uint_t i=0; i<param_map.size();++i)
		{
			ret(i)=(*it).first;
			++it;
		}
		return ret;
	}

	inline void clear()
	{
		this->param_map.clear();
	};
};

class CGPbase {
protected:

	MatrixXd X;    //training inputs
	MatrixXd Y;    //training targets
	//VectorXd meanY; //mean of training targets
	
	//cached GP-parameters:
	MatrixXd K;
	MatrixXd Kinv;
	MatrixXd KinvY;
	MatrixXd DKinv_KinvYYKinv;

	Eigen::LDLT<gpmix::MatrixXd> cholK;
	CGPHyperParams params;

	ACovarianceFunction& covar;//Covariance function
	ALikelihood& lik;          //likelihood model
	
	virtual MatrixXd getK();
	virtual MatrixXd getKinv();
	virtual MatrixXd getKinvY();
	virtual Eigen::LDLT<gpmix::MatrixXd> getCholK();
	virtual MatrixXd getDKinv_KinvYYKinv();

	virtual void clearCache();

	virtual MatrixXd LMLgrad_covar();
	virtual MatrixXd LMLgrad_lik();

public:
	CGPbase(ACovarianceFunction& covar, ALikelihood& lik);
	virtual ~CGPbase();

//TODO: add interface that is suitable for optimizer
// virtual double LML(double* params);
// virtual void LML(double* params, double* gradients);
	virtual void set_data(MatrixXd& X, MatrixXd& Y, CGPHyperParams& hyperparams);

	//virtual void set_params(CGPHyperParams& hyperparams);

	virtual float_t LML();        //the log-likelihood (+ log-prior)
	virtual CGPHyperParams LMLgrad();   //the gradient of the log-likelihood (+ log-prior)

	inline uint_t get_samplesize(){return this->Y.rows();} //get the number of training data samples
	inline uint_t get_input_dimension(){return this->X.cols();} //get the number of training data samples
	inline uint_t get_target_dimension(){return this->Y.cols();} //get the dimension of the target data
	virtual MatrixXd predictMean(MatrixXd& Xstar);
	virtual MatrixXd predictVar(MatrixXd& Xstar);
};

} /* namespace gpmix */
#endif /* GP_BASE_H_ */
