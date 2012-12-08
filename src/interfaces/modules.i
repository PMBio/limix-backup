%shared_ptr(limix::CVqtl)
%shared_ptr(limix::CMultiTraitVQTL)
%shared_ptr(limix::CVarianceDecomposition)
%shared_ptr(limix::AVarianceTerm)
%shared_ptr(limix::CSingleTraitVarianceTerm)
%shared_ptr(limix::CCategorialTraitVarianceTerm)

//vector template definitions
namespace std {
   %template(VectorXiVec) vector<VectorXi>;
};
