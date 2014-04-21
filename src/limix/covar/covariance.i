namespace limix {
//ACovarianceFunction

%ignore ACovarianceFunction::K;
%ignore ACovarianceFunction::Kdiag;
%ignore ACovarianceFunction::Kcross_diag;
%ignore ACovarianceFunction::Kdiag_grad_X;
%ignore ACovarianceFunction::Kgrad_X;
%ignore ACovarianceFunction::Kcross;
%ignore ACovarianceFunction::Kgrad_param;
%ignore ACovarianceFunction::Khess_param;
%ignore ACovarianceFunction::Kcross_grad_X;

%ignore ACovarianceFunction::getParams;
%ignore ACovarianceFunction::getParamMask;
%ignore ACovarianceFunction::getX;

//rename argout versions for python; this overwrites the C++ convenience functions
%rename(K) ACovarianceFunction::aK;
%rename(Kdiag) ACovarianceFunction::aKdiag;
%rename(Kdiag_grad_X) ACovarianceFunction::aKdiag_grad_X;
%rename(Kgrad_X) ACovarianceFunction::aKgrad_X;
%rename(Kcross) ACovarianceFunction::aKcross;
%rename(Kcross_diag) ACovarianceFunction::aKcross_diag;
%rename(Kgrad_param) ACovarianceFunction::aKgrad_param;
%rename(Khess_param) ACovarianceFunction::aKhess_param;
%rename(Kcross_grad_X) ACovarianceFunction::aKcross_grad_X;

%rename(getParams) ACovarianceFunction::agetParams;
%rename(getParamMask) ACovarianceFunction::agetParamMask;
%rename(getX) ACovarianceFunction::agetX;
%rename(getParamBounds) ACovarianceFunction::agetParamBounds;
%rename(getParamBounds0) ACovarianceFunction::agetParamBounds0;
    
%rename(Khess_param_num) ACovarianceFunction::aKhess_param_num;

}

//raw include
%include "limix/covar/covariance.h"