/*
 * likelihood.h
 *
 *  Created on: Nov 11, 2011
 *      Author: clippert
 */

#ifndef LIKELIHOOD_H_
#define LIKELIHOOD_H_

#include <gpmix/types.h>
#include <gpmix/covar/covariance.h>


namespace gpmix {


typedef VectorXd LikParams;

class ALikelihood {
	//indicator if the class is synced with the cache
protected:
	bool insync;
	LikParams params;
	muint_t numberParams;
	CovarInput X;

public:
	ALikelihood(const muint_t numberParams);
	virtual ~ALikelihood();

	//get the Vector of hyperparameters
	inline void getParams(CovarParams* out){(*out) = params;};
	//set the parameters to a new value.
	virtual void setParams(LikParams& params);
	//get the X
	inline void getX(CovarInput* Xout) const { (*Xout) = this->X;}
	//getDimX
	inline muint_t getDimX() const {return (muint_t)(this->X.cols());};
	//number of parameters:
	inline muint_t getNumberParams() const {return this->numberParams;};


	//check if object is  insync with cache
	inline bool isInSync() const {return insync;}
	//indicate that the cache has been cleared and is synced again
	inline void makeSync() { insync = true;}

	//pure virtual functions that need to be overwritten
	virtual void applyToK(MatrixXd& K) const = 0;
	virtual void K(MatrixXd* out) const = 0;
	virtual void Kdiag(VectorXd* out) const =0;
	virtual void Kgrad_params(MatrixXd* out, const muint_t row) const = 0;

#ifndef SWIG
	inline CovarInput getX() const {return this->X;}
	inline LikParams getParams() const {return params;}
	virtual string getName() const = 0;

	inline virtual MatrixXd K();
	inline virtual VectorXd Kdiag();
	inline virtual MatrixXd Kgrad_params(const muint_t row);
#endif
};


#ifndef SWIG
inline MatrixXd ALikelihood::K()
{
	MatrixXd RV;
	K(&RV);
	return RV;
}
inline VectorXd ALikelihood::Kdiag()
{
	VectorXd RV;
	Kdiag(&RV);
	return RV;
}
inline MatrixXd ALikelihood::Kgrad_params(const muint_t row)
{
	MatrixXd RV;
	Kgrad_params(&RV,row);
	return RV;
};
#endif


class CLikNormalIso : public ALikelihood {

public:
	CLikNormalIso();
	~CLikNormalIso();

	//pure virtual functions that need to be overwritten
	virtual void applyToK(MatrixXd& K) const = 0;
	virtual void K(MatrixXd* out) const = 0;
	virtual void Kdiag(VectorXd* out) const =0;
	virtual void Kgrad_params(MatrixXd* out, const muint_t row) const = 0;

#ifndef SWIG
	string getName() const {return "LikNormalIso";};
#endif
};





} /* namespace gpmix */
#endif /* LIKELIHOOD_H_ */
