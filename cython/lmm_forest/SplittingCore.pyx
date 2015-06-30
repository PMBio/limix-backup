import scipy.stats as st
import numpy as NP
cimport numpy as NP
from libcpp cimport bool
from numpy cimport int64_t

cdef extern from "numpy/arrayobject.h":
    ctypedef int intp
    ctypedef extern class numpy.ndarray [object PyArrayObject]:
        cdef char *data
        cdef int nd
        cdef intp *dimensions
        cdef intp *strides
        cdef int flags


cdef extern from "types.h":
    ctypedef double mfloat_t
    ctypedef int64_t mint_t


    cdef cppclass MatrixXd:
        MatrixXd()
        int rows()
        int cols()

    cdef cppclass MMatrixXd:
        MMatrixXd(mfloat_t *,int,int)
        int rows()
        int cols()

    cdef cppclass MMatrixXdRM:
        MMatrixXdRM(mfloat_t *,int,int)
        int rows()
        int cols()

    cdef cppclass MMatrixXi:
        MMatrixXi(mint_t *,int,int)
        int rows()
        int cols()


##conversion for numpy arrays
def ndarrayF64toC(ndarray A):
    #transform numpy object to enforce Fortran contiguous byte order
    #(meaining column-first order for Cpp interfacing)
    return NP.asarray(A, order="F")

cdef extern from "SplittingCore.h":
    void c_ml_beta "ml_beta" (MatrixXd* out, MMatrixXd* UTy, MMatrixXd* UTX, MMatrixXd* S, mfloat_t delta)
    #void c_ml_sigma "ml_sigma" (MatrixXd* out, MMatrixXd* UTy, MMatrixXd* UTX, MMatrixXd* S, mfloat_t delta)
    mfloat_t c_estimate_bias "estimate_bias" (
                                              MMatrixXd* UTy,
                                              MMatrixXd* UT,
                                              MMatrixXd* S,
                                              mfloat_t delta)
    void c_best_split_full_model "best_split_full_model" (
                                                          mint_t* m_best,
                                                          mfloat_t* s_best,
                                                          mfloat_t* left_mean,
                                                          mfloat_t* right_mean,
                                                          mfloat_t* ll_score,
                                                          MMatrixXdRM* X,
                                                          MMatrixXd* UTy,
                                                          MMatrixXd* C,
                                                          MMatrixXd* S,
                                                          MMatrixXd* U,
                                                          MMatrixXi* noderange,
                                                          mfloat_t delta)

    void c_predict_rec "predict_rec" (mfloat_t* response,
                              mint_t root,
                              MMatrixXi* tree_nodes,
                              MMatrixXi* left_children,
                              MMatrixXi* right_children,
                              MMatrixXi* best_predictor,
                              MMatrixXd* mean,
                              MMatrixXd* splitting_value,
                              MMatrixXd* X,
                              mfloat_t depth)

    void c_predict "predict" (MMatrixXd* response,
                              MMatrixXi* tree_nodes,
                              MMatrixXi* left_children,
                              MMatrixXi* right_children,
                              MMatrixXi* best_predictor,
                              MMatrixXd* mean,
                              MMatrixXd* splitting_value,
                              MMatrixXd* X,
                              mfloat_t depth)

    void c_test "test" (MatrixXd* out, MMatrixXd* XR,MMatrixXd* yR)
    void c_copy "copy" (MMatrixXd* m_out, MatrixXd* m_in)

def estimate_bias(ndarray UTy, ndarray UT, ndarray S, delta):
    UTy = ndarrayF64toC(UTy)
    UT = ndarrayF64toC(UT)
    S = ndarrayF64toC(S)

    cdef MMatrixXd* mUTy = new MMatrixXd(<double* > UTy.data,UTy.dimensions[0],UTy.dimensions[1])
    cdef MMatrixXd* mUT = new MMatrixXd(<double* > UT.data,UT.dimensions[0],UT.dimensions[1])
    cdef MMatrixXd* mS = new MMatrixXd(<double* > S.data,S.dimensions[0],1)
    return c_estimate_bias(mUTy, mUT, mS, delta)

def best_split_full_model(ndarray X,
                          ndarray UTy,
                          ndarray C,
                          ndarray S,
                          ndarray U,
                          ndarray noderange,
                          mfloat_t delta):

    # Note, X is also row major in Cpp
    UTy = ndarrayF64toC(UTy)
    U = ndarrayF64toC(U)
    S = ndarrayF64toC(S)
    C = ndarrayF64toC(C)

    cdef MMatrixXdRM* mX = new MMatrixXdRM(<double*> X.data,X.dimensions[0],X.dimensions[1])
    cdef MMatrixXd* mUTy = new MMatrixXd(<double*> UTy.data,UTy.dimensions[0],1)
    cdef MMatrixXd* mC = new MMatrixXd(<double*> C.data,C.dimensions[0],C.dimensions[1])
    cdef MMatrixXd* mS = new MMatrixXd(<double*> S.data,S.dimensions[0],1)
    cdef MMatrixXd* mU = new MMatrixXd(<double*> U.data,U.dimensions[0],U.dimensions[1])
    cdef MMatrixXi* mnoderange = new MMatrixXi(<mint_t* > noderange.data,noderange.dimensions[0],1)

    cdef NP.ndarray m_best=NP.zeros([1,1],dtype=NP.int64)
    cdef NP.ndarray s_best=NP.zeros([1,1],dtype=NP.float)
    cdef NP.ndarray left_mean=NP.zeros([1,1],dtype=NP.float)
    cdef NP.ndarray right_mean=NP.zeros([1,1],dtype=NP.float)
    cdef NP.ndarray ll_score=NP.zeros([1,1],dtype=NP.float)
    m_best[0,0] = -1
    s_best[0,0] = 0.0
    left_mean[0,0] = 0.0
    right_mean[0,0] = 0.0
    ll_score[0,0] = 0.0
    #References to objects passed to cpp function
    cdef mint_t* dm_best = <mint_t*> m_best.data
    cdef double* ds_best = <double*> s_best.data
    cdef double* dleft_mean = <double*> left_mean.data
    cdef double* dright_mean = <double*> right_mean.data
    cdef double* dll_score = <double*> ll_score.data

    c_best_split_full_model(
                            dm_best,
                            ds_best,
                            dleft_mean,
                            dright_mean,
                            dll_score,
                            mX,
                            mUTy,
                            mC,
                            mS,
                            mU,
                            mnoderange,
                            delta)

    return m_best[0,0], s_best[0,0], left_mean[0,0], right_mean[0,0], ll_score[0,0]

#############################################
### routines for making prediction faster ###
#############################################



def predict_rec(ndarray tree_nodes,
            ndarray left_children,
            ndarray right_children,
            ndarray best_predictor,
            ndarray mean,
            ndarray splitting_value,
            ndarray X,
            mfloat_t depth):

    X = ndarrayF64toC(X)
    cdef MMatrixXi* mtree_nodes = new MMatrixXi(<mint_t*> tree_nodes.data, tree_nodes.dimensions[0],1)
    cdef MMatrixXi* mleft_children = new MMatrixXi(<mint_t*> left_children.data, left_children.dimensions[0],1)
    cdef MMatrixXi* mright_children = new MMatrixXi(<mint_t*> right_children.data, right_children.dimensions[0],1)
    cdef MMatrixXi* mbest_predictor = new MMatrixXi(<mint_t*> best_predictor.data, best_predictor.dimensions[0],1)
    cdef MMatrixXd* mmean = new MMatrixXd(<double*> mean.data, mean.dimensions[0],1)
    cdef MMatrixXd* msplitting_value = new MMatrixXd(<double*> splitting_value.data, splitting_value.dimensions[0],1)
    cdef MMatrixXd* mX = new MMatrixXd(<double*> X.data,X.dimensions[0],X.dimensions[1])

    cdef NP.ndarray response=NP.zeros([1,1])
    cdef double* dresponse = <double*> response.data
    cdef mint_t root = 0
    c_predict_rec(dresponse, root, mtree_nodes, mleft_children, mright_children, mbest_predictor, mmean, msplitting_value, mX, depth)
    print 'passed c prediction function'
    return response[0,0]

def predict(ndarray response,
            ndarray tree_nodes,
            ndarray left_children,
            ndarray right_children,
            ndarray best_predictor,
            ndarray mean,
            ndarray splitting_value,
            ndarray X,
            mfloat_t depth):

    X = ndarrayF64toC(X)
    cdef MMatrixXi* mtree_nodes = new MMatrixXi(<mint_t*> tree_nodes.data, tree_nodes.dimensions[0],1)
    cdef MMatrixXi* mleft_children = new MMatrixXi(<mint_t*> left_children.data, left_children.dimensions[0],1)
    cdef MMatrixXi* mright_children = new MMatrixXi(<mint_t*> right_children.data, right_children.dimensions[0],1)
    cdef MMatrixXi* mbest_predictor = new MMatrixXi(<mint_t*> best_predictor.data, best_predictor.dimensions[0],1)
    cdef MMatrixXd* mmean = new MMatrixXd(<double*> mean.data, mean.dimensions[0],1)
    cdef MMatrixXd* msplitting_value = new MMatrixXd(<double*> splitting_value.data, splitting_value.dimensions[0],1)
    cdef MMatrixXd* mX = new MMatrixXd(<double*> X.data,X.dimensions[0],X.dimensions[1])

    #cdef NP.ndarray response=NP.zeros([X.dimensions[0],1])
    cdef MMatrixXd* mresponse = new MMatrixXd(<double*> response.data,response.dimensions[0],1)
    c_predict(mresponse, mtree_nodes, mleft_children, mright_children, mbest_predictor, mmean, msplitting_value, mX, depth)
    return response


###test case####
def test(ndarray Xr,ndarray yR,delta,ndarray S):
    #makre sure numpy arrays are contiguous
    Xr = ndarrayF64toC(Xr)
    yR = ndarrayF64toC(yR)
    S  = ndarrayF64toC(S)

    #map to Eigen
    cdef MMatrixXd* mXr = new MMatrixXd(<double* > Xr.data,Xr.dimensions[0],Xr.dimensions[1])
    cdef MMatrixXd* myR = new MMatrixXd(<double* > yR.data,yR.dimensions[0],yR.dimensions[1])
    cdef MMatrixXd* mS = new MMatrixXd(<double* > S.data,S.dimensions[0],S.dimensions[1])

    cdef MatrixXd* cout = new MatrixXd()
    c_test(cout,mXr,myR);
    #copy results back to python structures
    cdef NP.ndarray pout  = NP.zeros([cout.rows(),cout.cols()],dtype=NP.float,order="F")
    cdef MMatrixXd* mpout = new MMatrixXd(<double* > pout.data,cout.rows(),cout.cols())
    c_copy(mpout,cout)
    return pout
