%shared_ptr(bost::enable_shared_from_this<CGPbase>)
%shared_ptr(std::map<std::string,MatrixXd>)
%shared_ptr(limix::CCovarianceFunctionCache)
%shared_ptr(limix::CGPCholCache)
%shared_ptr(limix::CGPHyperParams)
%shared_ptr(limix::CGPbase)
%shared_ptr(limix::CGPkronecker)
%shared_ptr(limix::CGPopt)
%shared_ptr(limix::CGPKroneckerCache)

//vector template definitions
namespace std {
   %template(MatrixXdVec) vector<MatrixXd>;
   %template(StringVec)   vector<string>;
   %template(StringMatrixMap) map<string,MatrixXd>;
};
