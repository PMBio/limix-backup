/*
 * ADataTerm.h
 *
 *  Created on: Jan 3, 2012
 *      Author: clippert
 */

#ifndef ADATATERM_H_
#define ADATATERM_H_

#include <gpmix/types.h>

namespace gpmix {

//rename argout operators for swig interface
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//ignore C++ versions
%ignore ACovarianceFunction::K;
%ignore ACovarianceFunction::Kdiag;
%ignore ACovarianceFunction::Kdiag_grad_X;
%ignore ACovarianceFunction::Kgrad_X;
%ignore ACovarianceFunction::Kcross;
%ignore ACovarianceFunction::Kgrad_param;
%ignore ACovarianceFunction::Kcross_grad_X;

%ignore ACovarianceFunction::getParams;
%ignore ACovarianceFunction::getX;

//rename argout versions for python; this overwrites the C++ convenience functions
%rename(K) ACovarianceFunction::aK;
%rename(Kdiag) ACovarianceFunction::aKdiag;
%rename(Kdiag_grad_X) ACovarianceFunction::aKdiag_grad_X;
%rename(Kgrad_X) ACovarianceFunction::aKgrad_X;
%rename(Kcross) ACovarianceFunction::aKcross;
%rename(Kgrad_param) ACovarianceFunction::aKgrad_param;
%rename(Kcross_grad_X) ACovarianceFunction::aKcross_grad_X;

%rename(getParams) ACovarianceFunction::agetParams;
%rename(getX) ACovarianceFunction::agetX;
#endif


class ADataTerm {
protected:
	MatrixXd Y;
	bool insync;

public:
	ADataTerm();
	ADataTerm(MatrixXd& Y);
	virtual ~ADataTerm();
	virtual inline void setParams(MatrixXd& params){};

	virtual void aGetParams(MatrixXd* outParams){};
	virtual inline MatrixXd getParams(){ MatrixXd outParams = MatrixXd(); aGetParams(&outParams); return outParams;	};
	virtual inline void setY(const MatrixXd& Y){
		this->insync = false;
		this->Y = Y;
	}

	virtual void aEvaluate(MatrixXd* Y);
	virtual void aGradY(MatrixXd* gradY);
	virtual void aGradParams(MatrixXd* gradParams);
	virtual void aSumJacobianGradParams(MatrixXd* sumJacobianGradParams);
	virtual void aSumLogJacobian(MatrixXd* sumJacobianGradParams);

	virtual inline MatrixXd getY(){return Y;}
	virtual inline MatrixXd evaluate() { MatrixXd ret = MatrixXd(); aEvaluate(&ret); return ret;};
	virtual inline MatrixXd gradY() { MatrixXd ret = MatrixXd(); aGradY(&ret); return ret;};
	virtual inline MatrixXd gradParams(){ MatrixXd ret = MatrixXd(); aGradParams(&ret); return ret;};
	virtual inline MatrixXd sumJacobianGradParams(){ MatrixXd ret = MatrixXd(); aSumJacobianGradParams(&ret); return ret;};
	virtual inline MatrixXd sumLogJacobian(){ MatrixXd ret = MatrixXd(); aSumLogJacobian(&ret); return ret;};
	bool isInSync() const;
	void makeSync();
};

} /* namespace gpmix */
#endif /* ADATATERM_H_ */
