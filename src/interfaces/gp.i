// Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
// All rights reserved.
//
// LIMIX is provided under a 2-clause BSD license.
// See license.txt for the complete license.

%shared_ptr(bost::enable_shared_from_this<CGPbase>)
%shared_ptr(std::map<std::string,MatrixXd>)
%shared_ptr(limix::CCovarianceFunctionCache)
%shared_ptr(limix::CCovarianceFunctionCacheOld)
%shared_ptr(limix::CGPCholCache)
%shared_ptr(limix::CGPHyperParams)
%shared_ptr(limix::CGPbase)
%shared_ptr(limix::CGPvarDecomp)
%shared_ptr(limix::CGPkronecker)
%shared_ptr(limix::CGPkronSum)
%shared_ptr(limix::CGPSum)
%shared_ptr(limix::CGPopt)
%shared_ptr(limix::CGPKroneckerCache)
%shared_ptr(limix::CGPkronSumCache)
%shared_ptr(limix::CGPSumCache)

//vector template definitions
namespace std {
   %template(MatrixXdVec) vector<MatrixXd>;
   %template(StringVec)   vector<string>;
   %template(StringMatrixMap) map<string,MatrixXd>;
};
