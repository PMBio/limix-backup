%shared_ptr(gpmix::CGPHyperParams)
%shared_ptr(gpmix::CGPbase)
%shared_ptr(gpmix::CGPkronecker)


//vector template definitions
namespace std {
   %template(MatrixXdVec) vector<MatrixXd>;
   %template(StringVec)   vector<string>;
   %template(StringMatrixMap) map<string,MatrixXd>;
};
