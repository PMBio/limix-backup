%shared_ptr(limix::CGPHyperParams)
%shared_ptr(limix::CGPbase)
%shared_ptr(limix::CGPkronecker)


//vector template definitions
namespace std {
   %template(MatrixXdVec) vector<MatrixXd>;
   %template(StringVec)   vector<string>;
   %template(StringMatrixMap) map<string,MatrixXd>;
};
