%shared_ptr(limix::CVqtl)
%shared_ptr(limix::CMultiTraitVQTL)
%shared_ptr(limix::AVarianceTerm)
%shared_ptr(limix::CSingleTraitTerm)
%shared_ptr(limix::CMultiTraitTerm)
%shared_ptr(limix::CVarianceDecomposition)

//vector template definitions
namespace std {
   %template(VectorXiVec) vector<VectorXi>;
};
