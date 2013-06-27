%shared_ptr(limix::CVqtl)
%shared_ptr(limix::CMultiTraitVQTL)
%shared_ptr(limix::CVarianceDecomposition)
%shared_ptr(limix::AVarianceTerm)
%shared_ptr(limix::CSingleTraitVarianceTerm)
%shared_ptr(limix::CCategorialTraitVarianceTerm)
%shared_ptr(limix::AVarianceTermN)
%shared_ptr(limix::CSingleTraitTerm)
%shared_ptr(limix::CMultiTraitTerm)
%shared_ptr(limix::CNewVarianceDecomposition)

//vector template definitions
namespace std {
   %template(VectorXiVec) vector<VectorXi>;
};
