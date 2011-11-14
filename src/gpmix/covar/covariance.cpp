/*
 * ACovariance.cpp
 *
 *  Created on: Nov 10, 2011
 *      Author: stegle
 */

#include "covariance.h"

namespace gpmix {

ACovarianceFunction::ACovarianceFunction(const uint_t dimensions)
{
	this->dimensions_i1 = dimensions;
	this->dimensions_i0 = 0;
}

ACovarianceFunction::ACovarianceFunction(const uint_t dimensions_i0,const uint_t dimensions_i1)
{
	this->dimensions_i0 = dimensions_i0;
	this->dimensions_i1 = dimensions_i1;
}

MatrixXd ACovarianceFunction::K(const CovarParams params, const CovarInput x1) const
{
	return K(params,x1,x1);
}


ACovarianceFunction::~ACovarianceFunction()
{
}

CovarInput ACovarianceFunction::getX(const CovarInput x) const
{
	return x.block(0,dimensions_i0,x.rows(),dimensions_i1);
}

bool ACovarianceFunction::dimension_is_target(const uint_t d) const
{
	return ((d>=this->dimensions_i0) & (d<this->dimensions_i1));
}

VectorXd ACovarianceFunction::Kdiag(CovarParams params,CovarInput x1) const
/*
 * Default implementation of diagional covariance operator
 */
{
	MatrixXd K = this->K(params,x1,x1);
	return K.diagonal();
}


//gradcheck functions for covaraince classes

bool check_covariance_Kgrad_theta(const ACovarianceFunction& covar,const CovarParams params, const CovarInput x,float_t relchange,float_t threshold)
{
	float_t RV=0;
	//copy of parameter vector
	CovarParams L = params;
	//dimensions
	for(int_t i=0;i<L.rows();i++)
	{
		float_t change = relchange*L(i);
		        change = max(change,1E-5);
		L(i) = params(i) + change;
		MatrixXd Lplus = covar.K(L,x);
		L(i) = params(i) - change;
		MatrixXd Lminus = covar.K(L,x);
		//numerical gradient
		MatrixXd diff_numerical  = (Lplus-Lminus)/(2.*change);
		//analytical gradient
		MatrixXd diff_analytical = covar.Kgrad_theta(params,x,i);
		RV += (diff_numerical-diff_analytical).squaredNorm();
	}

	return (RV<threshold);
}


bool check_covariance_Kgrad_x(const ACovarianceFunction& covar,const CovarParams params, const CovarInput x,float_t relchange,float_t threshold)
{
return true;
} /* namespace gpmix */



/*
 * def grad_check_Kx(K,logtheta,x0,dimensions=None):
    """perform grad check with respect to input x"""
    L=0;
    x1 = x0.copy()
    n = x1.shape[0]
    if dimensions is None:
        dimensions = SP.arange(x0.shape[1])
    nd = len(dimensions)
    diff = SP.zeros([n,nd,n,n])
    for i in xrange(n):
        for iid in xrange(nd):
            d = dimensions[iid]
            change = relchange*x0[i,d]
            change = max(change,1E-5)
            x1[i,d] = x0[i,d] + change
            Lplus = K.K(logtheta,x1,x1)
            x1[i,d] = x0[i,d] - change
            Lminus = K.K(logtheta,x1,x1)
            x1[i,d] = x0[i,d]

            diff[i,iid,:,:] = (Lplus-Lminus)/(2.*change)
    #ana
    ana = SP.zeros([n,nd,n,n])
    ana2 = SP.zeros([n,nd,n,n])
    for iid in xrange(nd):
        d = dimensions[iid]
        dKx = K.Kgrad_x(logtheta,x1,x1,d)
        #dKx_diag = K.Kgrad_xdiag(logtheta,x1,d)
        #dKx.flat[::(dKx.shape[1] + 1)] = dKx_diag
        for iin in xrange(n):
            dKxn = SP.zeros([n, n])
            dKxn[iin, :] = dKx[iin, :]
            dKxn[:, iin] += dKx[iin, :]
            ana[iin,iid,:,:] = dKxn

    delta = (ana -diff)/(diff+1E-10)
    print "delta %.2f" % SP.absolute(delta).max()
    for d in xrange(nd):
        pylab.close("all")
        pylab.figure()
        pylab.pcolor(ana[:,d,:,:].sum(0))
        pylab.title("analytical")
        pylab.colorbar()

        pylab.figure()
        pylab.pcolor(diff[:,d,:,:].sum(0))
        pylab.title("numerical")
        pylab.colorbar()
        import pdb;pdb.set_trace()
    pass
 */

